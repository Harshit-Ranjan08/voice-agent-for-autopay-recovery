import os
import io
import uuid
import base64
import time
import wave
import hashlib
import logging
import httpx
from dotenv import load_dotenv
 
load_dotenv()
 
# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("services.sarvam")
 
# ==========================================
# ENVIRONMENT CONFIG
# ==========================================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech")
 
# ==========================================
# HYPERPARAMETERS (Best for NDR Calling)
# ==========================================
 
# --- TTS Hyperparameters ---
# model: bulbul:v3 - latest and most natural model from Sarvam
TTS_MODEL = "bulbul:v3"
 
# speaker_hindi: "priya" - clear female voice for Bulbul v3 Hindi
SPEAKER_HINDI = "priya"
 
# speaker_english: "simran" - clear female voice for Bulbul v3 English
SPEAKER_ENGLISH = "simran"
 
# speech_sample_rate: 8000 Hz is the telephony standard (used by Exotel/IVR systems)
# Using 8000 avoids resampling overhead and works natively with phone lines
TTS_SAMPLE_RATE = 8000
 
# audio_format: "wav" - universally compatible, no decoding artifacts,
# and native format for Exotel's <Play> XML tag
TTS_AUDIO_FORMAT = "wav"
 
# speech_speed: 1.0 = normal. Slightly slower (0.9) improves clarity for IVR
TTS_SPEECH_SPEED = 1.0
 
# enable_preprocessing: True - handles numbers, dates, and mixed Hindi-English text better
TTS_ENABLE_PREPROCESSING = True
 
# loudness: NOT SENT. Confirmed via production 400s on 2026-08-20 that Sarvam's
# Bulbul V3 model rejects the "loudness" (and "pitch") parameters outright:
# {"error":{"message":"Pitch and loudness parameters are currently not
# supported for the Bulbul V3 model. Please do not pass these values."}}
# Kept here (unused) so it's easy to re-enable if/when Sarvam adds support.
TTS_LOUDNESS = 1.2
 
# max_chars: Sarvam's documented REST limit. Validate/truncate client-side
# instead of letting a long generated sentence come back as an opaque 4xx.
TTS_MAX_CHARS = 2500
 
# --- STT Hyperparameters ---
# model: saaras:v3 - latest best-in-class multilingual model (Hindi + English + code-mix)
STT_MODEL = "saaras:v3"
 
# language_code: "hi-IN" - supports Hindi and Hindi-English codemix (most NDR customers)
# The model auto-detects English as well, so this is a safe default for mixed speech
STT_LANGUAGE_CODE = "hi-IN"
 
# mode: "transcribe" - direct transcription mode (no translation)
# Use "codemix" if the customer frequently mixes Hindi + English in a single sentence
STT_MODE = "codemix"
 
# Once a customer explicitly switches languages mid-call (agent.current_language
# flips to "English"), hint STT with the matching language_code so pure-English
# utterances aren't transcribed against a Hindi-primary prior. Pass
# language="English" into speech_to_text() to use this.
STT_LANGUAGE_CODE_BY_LANGUAGE = {
    "hindi": "hi-IN",
    "english": "en-IN",
}
 
# --- Network / reliability tuning ---
# A live phone call is the worst place for a generic 30s blanket timeout --
# the customer just sits in silence. Fail faster and retry once instead of
# blocking on a single slow attempt. Tune these against your own measured
# p95 latency from Sarvam; these are conservative starting points, not
# universal constants.
CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 8.0
WRITE_TIMEOUT = 8.0
POOL_TIMEOUT = 3.0
MAX_RETRIES = 1                    # 1 retry => 2 attempts total per call
RETRY_BACKOFF_SECONDS = 0.4        # multiplied by attempt number
 
# How long cached TTS files are kept before cleanup_tts_cache() removes them.
# Most NDR pitches are interpolated with per-customer data (name, item, COD
# amount...) so most cache entries are effectively one-time-use; left
# unchecked, static/ grows without bound over weeks of campaign calling.
TTS_CACHE_MAX_AGE_DAYS = 14
 
# Optional: pre-recorded apology clips to fall back to if Sarvam TTS is
# unreachable after retries, so a call can still end gracefully instead of
# raising mid-turn. Purely additive -- if these files don't exist, behavior
# is identical to before (the exception just propagates).
FALLBACK_AUDIO = {
    "Hindi": "static/fallback_apology_hindi.wav",
    "English": "static/fallback_apology_english.wav",
}
 
# Ensure static directory exists to store TTS audio files served by FastAPI
os.makedirs("static", exist_ok=True)
 
 
class SarvamAPIError(Exception):
    """Raised when a Sarvam API call ultimately fails after retries/backoff."""
 
 
# Global persistent HTTP client with connection pooling for low-latency API
# calls. Timeouts are split (connect/read/write/pool) instead of one blanket
# value so a slow DNS/connect doesn't eat the same budget as a slow response.
_http_client = httpx.Client(
    timeout=httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    ),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
)
 
 
def validate_sarvam_config() -> None:
    """
    Call once from FastAPI's startup event. A missing API key previously only
    surfaced on the first live customer call (mid-conversation, as a raised
    ValueError); this lets a bad deployment fail immediately at boot instead.
    """
    missing = [name for name, val in [
        ("SARVAM_API_KEY", SARVAM_API_KEY),
        ("SARVAM_STT_URL", SARVAM_STT_URL),
        ("SARVAM_TTS_URL", SARVAM_TTS_URL),
    ] if not val]
    if missing:
        raise RuntimeError(f"Missing required Sarvam config: {', '.join(missing)}")
    logger.info("[SARVAM] Config validated OK.")
 
 
def close_http_client() -> None:
    """Call from FastAPI's shutdown event to release pooled connections cleanly."""
    _http_client.close()
 
 
def _request_with_retry(method: str, url: str, **kwargs) -> httpx.Response:
    """
    Shared retry/backoff wrapper for every outbound Sarvam call (TTS POST,
    STT POST, and STT recording-URL download GET). Retries network errors,
    5xx, and 429 (respecting Retry-After); returns immediately on any other
    status so callers can raise a precise error instead of retrying a 400.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = _http_client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            logger.warning(f"[SARVAM] network error on {method} {url} (attempt {attempt}/{MAX_RETRIES + 1}): {exc}")
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise SarvamAPIError(f"{method} {url} unreachable after {attempt} attempt(s): {exc}") from exc
 
        if response.status_code == 429 and attempt <= MAX_RETRIES:
            retry_after = float(response.headers.get("Retry-After", RETRY_BACKOFF_SECONDS))
            logger.warning(f"[SARVAM] 429 rate limited on {method} {url}, backing off {retry_after:.1f}s (attempt {attempt})")
            time.sleep(retry_after)
            continue
 
        if response.status_code >= 500 and attempt <= MAX_RETRIES:
            logger.warning(f"[SARVAM] {response.status_code} server error on {method} {url} (attempt {attempt}), retrying")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
 
        return response
 
    raise SarvamAPIError("Sarvam retry loop exited unexpectedly")
 
 
def _empty_text_fallback(language: str) -> str:
    return "धन्यवाद।" if (language or "Hindi").lower() == "hindi" else "Thank you."
 
 
def _validate_wav_format(filepath: str) -> None:
    """
    main.py's Exotel streamer blindly strips the first 44 bytes of this file
    and pipes the rest out as raw 16-bit/8kHz/mono PCM. If Sarvam ever changes
    its default WAV output, that assumption breaks silently and the customer
    just hears garbled/fast/slow audio. Check it once here so a mismatch shows
    up in logs instead of a support ticket.
    """
    try:
        with wave.open(filepath, "rb") as wf:
            channels, sampwidth, framerate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        if (channels, sampwidth, framerate) != (1, 2, TTS_SAMPLE_RATE):
            logger.warning(
                f"[SARVAM TTS] Unexpected WAV format in {filepath}: "
                f"channels={channels} sampwidth={sampwidth} framerate={framerate} "
                f"(expected 1/2/{TTS_SAMPLE_RATE}) — telephony streaming may sound wrong."
            )
    except Exception as exc:
        logger.warning(f"[SARVAM TTS] Could not validate WAV format for {filepath}: {exc}")
 
 
def cleanup_tts_cache(max_age_days: int = TTS_CACHE_MAX_AGE_DAYS, cache_dir: str = "static") -> int:
    """
    Deletes cached tts_cache_*.wav files older than max_age_days. The MD5
    cache has no eviction of its own; wire this into apscheduler (already a
    project dependency) as a daily job, e.g.:
 
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(cleanup_tts_cache, "cron", hour=3)
        scheduler.start()
 
    Returns the number of files deleted.
    """
    if not os.path.isdir(cache_dir):
        return 0
    cutoff = time.time() - (max_age_days * 86400)
    deleted = 0
    for fname in os.listdir(cache_dir):
        if not fname.startswith("tts_cache_") or not fname.endswith(".wav"):
            continue
        fpath = os.path.join(cache_dir, fname)
        try:
            if os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                deleted += 1
        except OSError as exc:
            logger.warning(f"[SARVAM TTS] Could not clean up {fpath}: {exc}")
    if deleted:
        logger.info(f"[SARVAM TTS] Cache cleanup removed {deleted} file(s) older than {max_age_days}d")
    return deleted
 
 
# ==========================================
# SARVAM TTS SERVICE
# ==========================================
 
class TTSService:
    @staticmethod
    def text_to_speech(
        text: str,
        language: str = "Hindi"
    ) -> str:
        """
        Converts a text string to speech using Sarvam Bulbul v3 with MD5 Disk Caching.
 
        Args:
            text: The text to convert to audio (max 2500 chars for REST API)
            language: "Hindi" or "English" (selects appropriate language code + speaker)
 
        Returns:
            filename (str): The generated or cached .wav filename in static/ directory
        """
        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY not found in environment variables.")
 
        text = (text or "").strip()
        if not text:
            logger.warning("[SARVAM TTS] Empty text passed in — substituting a generic line instead of calling the API.")
            text = _empty_text_fallback(language)
 
        if len(text) > TTS_MAX_CHARS:
            logger.warning(f"[SARVAM TTS] Text length {len(text)} exceeds the {TTS_MAX_CHARS}-char API limit, truncating.")
            text = text[:TTS_MAX_CHARS]
 
        # Select language code and speaker based on detected language
        if language.lower() == "hindi":
            language_code = "hi-IN"
            speaker = SPEAKER_HINDI
        else:
            language_code = "en-IN"
            speaker = SPEAKER_ENGLISH
 
        # Check MD5 cache to bypass API call if text has been generated previously
        cache_key = hashlib.md5(f"{text}:{language_code}:{speaker}:{TTS_MODEL}".encode("utf-8")).hexdigest()
        cached_filename = f"tts_cache_{cache_key}.wav"
        cached_filepath = os.path.join("static", cached_filename)
 
        if os.path.exists(cached_filepath) and os.path.getsize(cached_filepath) > 100:
            logger.info(f"[SARVAM TTS] CACHE HIT (0ms) | file={cached_filepath} | text='{text[:40]}...'")
            return cached_filename
 
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
 
        payload = {
            "text": text,
            "target_language_code": language_code,
            "speaker": speaker,
            "model": TTS_MODEL,
            "speech_sample_rate": TTS_SAMPLE_RATE,        # 8000 Hz: telephony standard
            "enable_preprocessing": TTS_ENABLE_PREPROCESSING, # handles numbers, dates, mixed text
            "speech_speed": TTS_SPEECH_SPEED,              # 1.0: normal speed
            "audio_format": TTS_AUDIO_FORMAT,              # wav: compatible with Exotel <Play>
            # NOTE: do NOT add "loudness" here -- Bulbul V3 rejects it with a 400.
            # See TTS_LOUDNESS comment above.
        }
 
        tts_start = time.perf_counter()
        logger.info(f"[SARVAM TTS] START | speaker={speaker} | language={language_code} | text_length={len(text)}")
 
        try:
            response = _request_with_retry("POST", SARVAM_TTS_URL, json=payload, headers=headers)
        except SarvamAPIError as exc:
            logger.error(f"[SARVAM TTS] all retries failed: {exc}")
            fallback = FALLBACK_AUDIO.get(language)
            if fallback and os.path.exists(fallback):
                logger.warning(f"[SARVAM TTS] serving static fallback audio instead: {fallback}")
                return os.path.basename(fallback)
            raise
 
        api_duration = time.perf_counter() - tts_start
        if response.status_code != 200:
            logger.error(f"[SARVAM TTS] ERROR | status={response.status_code} | duration={api_duration:.3f}s | body={response.text}")
            raise Exception(f"Sarvam TTS failed with status {response.status_code}: {response.text}")
 
        logger.info(f"[SARVAM TTS] API RESPONSE | duration={api_duration:.3f}s")
 
        data = response.json()
        if "audios" not in data or not data["audios"]:
            logger.error("[SARVAM TTS] response missing 'audios' field.")
            raise Exception("Sarvam TTS response is empty or missing 'audios' key.")
 
        try:
            audio_bytes = base64.b64decode(data["audios"][0])
        except Exception as exc:
            raise Exception(f"Sarvam TTS returned unparseable audio data: {exc}") from exc
 
        # Write atomically: two concurrent calls that land on the same
        # cache_key (very common for boilerplate closing/greeting lines
        # during a multi-call campaign, since worker threads run in
        # parallel via run_in_executor) must never let a reader see a
        # partially-written file.
        tmp_path = f"{cached_filepath}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        os.replace(tmp_path, cached_filepath)
 
        _validate_wav_format(cached_filepath)
 
        file_saved_duration = time.perf_counter() - tts_start
        logger.info(f"[SARVAM TTS] FILE SAVED | filepath={cached_filepath} | bytes={len(audio_bytes)} | duration={file_saved_duration:.3f}s")
        return cached_filename
 
 
# ==========================================
# SARVAM STT SERVICE
# ==========================================
 
class STTService:
    @staticmethod
    def speech_to_text(audio_input, language: str = "Hindi") -> str:
        """
        Transcribes audio (recording URL or raw bytes) to text using Sarvam Saaras v3.
 
        Args:
            audio_input: Either:
                - A URL string (http/https) pointing to an audio file (e.g., Exotel recording URL)
                - Raw bytes of an audio file (WAV, MP3, OGG)
                - A file-like object (BytesIO)
            language: "Hindi" (default) or "English" -- pass the call's current
                current_language so STT hints the matching language_code
                instead of always assuming Hindi-primary codemix. Safe to
                omit; defaults to the original hi-IN/codemix behavior.
 
        Returns:
            transcript (str): The transcribed text. Returns empty string if no speech detected.
 
        Raises:
            ValueError: If SARVAM_API_KEY is missing or audio_input is invalid
            Exception: If Sarvam API returns a non-200 response
        """
        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY not found in environment variables.")
 
        audio_file_like = None
 
        # Handle URL input: download the audio file first (reuses the pooled
        # client + retry/backoff instead of spinning up a fresh short-lived
        # httpx.Client per call, which skipped connection reuse entirely)
        if isinstance(audio_input, str) and (audio_input.startswith("http://") or audio_input.startswith("https://")):
            logger.info(f"Sarvam STT: downloading audio from URL: {audio_input}")
            download_res = _request_with_retry("GET", audio_input)
            if download_res.status_code != 200:
                raise Exception(f"Failed to download audio from URL: {audio_input} (status {download_res.status_code})")
            audio_file_like = io.BytesIO(download_res.content)
            logger.info(f"Sarvam STT: downloaded {len(download_res.content)} bytes")
 
        elif isinstance(audio_input, bytes):
            audio_file_like = io.BytesIO(audio_input)
            logger.info(f"Sarvam STT: using raw bytes input ({len(audio_input)} bytes)")
 
        elif hasattr(audio_input, "read"):
            audio_file_like = audio_input
            logger.info("Sarvam STT: using file-like object input")
 
        else:
            raise ValueError(f"Invalid audio_input type: {type(audio_input)}. Expected URL string, bytes, or file-like object.")
 
        headers = {
            "api-subscription-key": SARVAM_API_KEY
        }
 
        files = {
            # Send as recording.wav; Sarvam auto-detects actual codec
            "file": ("recording.wav", audio_file_like, "audio/wav")
        }
 
        language_code = STT_LANGUAGE_CODE_BY_LANGUAGE.get((language or "Hindi").lower(), STT_LANGUAGE_CODE)
 
        data = {
            "model": STT_MODEL,             # saaras:v3: best Hindi/English model
            "language_code": language_code, # hi-IN by default; en-IN once customer has switched
            "mode": STT_MODE,               # codemix: handles Hinglish speech naturally
            "with_timestamps": "false",     # not needed for phone call transcription
            "with_diarization": "false",    # only 1 speaker (customer) per segment
        }
 
        stt_start = time.perf_counter()
        logger.info(f"[SARVAM STT] START | model={STT_MODEL} | language={language_code} | mode={STT_MODE}")
 
        response = _request_with_retry("POST", SARVAM_STT_URL, headers=headers, files=files, data=data)
 
        api_duration = time.perf_counter() - stt_start
 
        if response.status_code != 200:
            logger.error(f"[SARVAM STT] ERROR | status={response.status_code} | duration={api_duration:.3f}s | body={response.text}")
            raise Exception(f"Sarvam STT failed with status {response.status_code}: {response.text}")
 
        logger.info(f"[SARVAM STT] API RESPONSE | duration={api_duration:.3f}s")
 
        res_data = response.json()
        transcript = res_data.get("transcript", "").strip()
 
        total_duration = time.perf_counter() - stt_start
        logger.info(f"[SARVAM STT] COMPLETE | duration={total_duration:.3f}s | text_length={len(transcript)} | transcript='{transcript[:80]}'")
        return transcript