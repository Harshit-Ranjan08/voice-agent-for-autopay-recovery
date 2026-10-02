import os
import io
import json
import uuid
import struct
import base64
import time
import math
import wave
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response, StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from database.connection import Base, engine, get_db, SessionLocal
from database.models import Campaign, AutopayCase, CallAttempt
import database.crud as crud
from services.parser import parse_autopay_file, parse_excel_report
from services.exotel import ExotelService
from services.sarvam import TTSService, STTService, validate_sarvam_config, close_http_client
from agent.agent import autopay_agent
from agent.state import AgentState

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

# ==========================================
# APP INITIALIZATION
# ==========================================
app = FastAPI(
    title="Razorpay Autopay Recovery Voice Agent API",
    description="Automated AI Voice Agent system for recovering failed autopay payments via telephony, WebSocket streaming, and live simulation.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bypasses ngrok free tier browser warning page for non-browser HTTP clients like Exotel
@app.middleware("http")
async def add_ngrok_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


def get_public_host_url(request: Request) -> str:
    """Returns the full public HTTPS URL of the server behind ngrok/proxy."""
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    forwarded_proto = request.headers.get("x-forwarded-proto") or "https"
    
    if forwarded_host and not forwarded_host.startswith("127.0.0.1") and not forwarded_host.startswith("localhost"):
        return f"{forwarded_proto}://{forwarded_host}".rstrip("/")
    return str(request.base_url).rstrip("/")


# Ensure static directory exists & mount it to serve generated TTS audio files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """
    Serves TTS WAV files with explicit ngrok headers so Exotel's media player can fetch audio.
    """
    file_path = os.path.join("static", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        headers={"ngrok-skip-browser-warning": "true"}
    )


active_sessions: Dict[str, Dict[str, Any]] = {}
pending_greetings: Dict[str, Dict[str, Any]] = {}
_global_webhook_domain: str = ""


# ==========================================
# VAD (Voice Activity Detection) HELPERS
# ==========================================
SPEECH_THRESHOLD = 300.0
SILENCE_THRESHOLD = 400.0
SILENCE_TIMEOUT_MS = 800
MAX_UTTERANCE_DURATION_S = 10.0
CHUNK_DURATION = 0.02


def _get_rms(pcm_data: bytes) -> float:
    count = len(pcm_data) // 2
    if count == 0:
        return 0.0
    try:
        samples = struct.unpack(f"<{count}h", pcm_data[:count * 2])
    except Exception:
        return 0.0
    return math.sqrt(sum(s ** 2 for s in samples) / count)


def _save_pcm_to_wav(pcm_data: bytes, file_path: str, sample_rate: int = 8000):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ==========================================
# EXOTEL VOICEBOT WEBSOCKET ENDPOINT
# ==========================================
@app.websocket("/ws/oxotol")
@app.websocket("/ws/exotel")
async def exotel_voicebot_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] Exotel Voicebot connected")

    stream_sid = ""
    call_sid = ""
    session_state: Optional[Dict[str, Any]] = None

    audio_buffer = bytearray()
    is_speaking = False
    silence_dur = 0.0
    no_speech_dur = 0.0
    last_media_ts = time.monotonic()
    first_media_logged = False
    utterance_start_ts: float = 0.0
    vad_muted_until: float = 0.0

    db = SessionLocal()
    loop = asyncio.get_event_loop()
    is_playing_audio = False

    async def _stream_wav_file(wav_path: str, is_greeting: bool = False):
        nonlocal is_playing_audio, is_speaking, silence_dur, no_speech_dur, last_media_ts, vad_muted_until
        is_playing_audio = True
        stream_start = time.perf_counter()
        try:
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            raw_pcm = wav_bytes[44:]
            chunk_size = 320
            start = loop.time()

            for idx, off in enumerate(range(0, len(raw_pcm), chunk_size)):
                chunk = raw_pcm[off:off + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = chunk + b"\x00" * (chunk_size - len(chunk))
                payload = base64.b64encode(chunk).decode()
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "stream_sid": stream_sid,
                    "media": {"payload": payload}
                }))

                sleep = start + (idx + 1) * CHUNK_DURATION - loop.time()
                if sleep > 0:
                    await asyncio.sleep(sleep)

        except (WebSocketDisconnect, RuntimeError):
            logger.info(f"[AUDIO OUT] Stream stopped: WebSocket disconnected | call_sid={call_sid}")
        except Exception as exc:
            logger.error(f"[AUDIO OUT] Stream error: {exc} | call_sid={call_sid}")
        finally:
            is_playing_audio = False
            is_speaking = False
            silence_dur = 0.0
            no_speech_dur = 0.0
            audio_buffer.clear()
            last_media_ts = time.monotonic()
            vad_muted_until = time.monotonic() + 0.8

    async def _tts_and_stream(text: str, language: str = "English"):
        audio_filename = await loop.run_in_executor(None, TTSService.text_to_speech, text, language)
        wav_path = os.path.join("static", audio_filename)
        await _stream_wav_file(wav_path, is_greeting=False)

    async def _run_agent_turn(user_text: str) -> bool:
        nonlocal session_state
        if not session_state:
            return True

        msgs = list(session_state.get("messages", []))
        msgs.append(HumanMessage(content=user_text))
        session_state["messages"] = msgs
        try:
            next_state = await loop.run_in_executor(None, autopay_agent.invoke, session_state)
            session_state = next_state
        except Exception as exc:
            logger.error(f"[AGENT] ERROR | call_sid={call_sid} | error={exc}")
            return False

        out_msgs = session_state.get("messages", [])
        bot_reply = out_msgs[-1].content if out_msgs else "Thank you!"
        lang = session_state.get("current_language", "English")
        await _tts_and_stream(bot_reply, lang)

        return session_state.get("current_step") == "completed"

    try:
        async for raw in websocket.iter_text():
            try:
                data = json.loads(raw)
            except Exception:
                continue

            event = data.get("event", "")

            if event == "start":
                start_data = data.get("start", {})
                call_sid = start_data.get("callSid") or start_data.get("call_sid") or data.get("callSid") or ""
                stream_sid = start_data.get("streamSid") or start_data.get("stream_sid") or data.get("streamSid") or ""
                custom_field = str(start_data.get("customField") or data.get("CustomField") or "").strip()

                logger.info(f"[WS] START EVENT | call_sid={call_sid} | stream_sid={stream_sid} | custom_field={custom_field}")

                pending = None
                if call_sid:
                    pending = pending_greetings.pop(call_sid, None)
                if not pending and custom_field:
                    pending = pending_greetings.pop(custom_field, None)
                if not pending and len(pending_greetings) == 1:
                    k, pending = next(iter(pending_greetings.items()))
                    pending_greetings.pop(k, None)

                from database.models import CallAttempt as CA, AutopayCase as AC

                if pending:
                    session_state = pending["agent_state"]
                    att_id = session_state.get("attempt_id")
                    if att_id:
                        att_obj = db.query(CA).filter(CA.id == att_id).first()
                        if att_obj:
                            att_obj.start_time = datetime.utcnow()
                            att_obj.call_status = "CONNECTED"
                            if call_sid:
                                att_obj.provider_call_id = call_sid
                            db.commit()
                    await _stream_wav_file(pending["wav_file"], is_greeting=True)
                else:
                    autopay_case = None
                    attempt = db.query(CA).filter(CA.provider_call_id == call_sid).first() if call_sid else None
                    if attempt:
                        autopay_case = crud.get_autopay_case(db=db, case_id=attempt.case_id)
                    if not autopay_case:
                        autopay_case = db.query(AC).filter(AC.agent_status.in_(["CALLING", "PENDING"])).order_by(AC.updated_at.desc()).first()

                    if autopay_case and not attempt:
                        existing_attempts = crud.get_call_attempts_for_case(db=db, case_id=autopay_case.id)
                        attempt = crud.create_call_attempt(
                            db=db,
                            case_id=autopay_case.id,
                            attempt_number=len(existing_attempts) + 1,
                            provider_call_id=call_sid or f"ws_{stream_sid}"
                        )
                    if attempt:
                        attempt.start_time = datetime.utcnow()
                        attempt.call_status = "CONNECTED"
                        db.commit()

                    customer_name = autopay_case.customer_name if autopay_case else "Customer"
                    service_type = autopay_case.service_type if autopay_case else "Service"
                    fallback_text = f"Hello, this is Arya calling from Razorpay regarding your {service_type} payment. Am I speaking with {customer_name}?"

                    session_state = {
                        "messages": [], "case_id": autopay_case.id if autopay_case else 0,
                        "attempt_id": attempt.id if attempt else None,
                        "customer_name": customer_name,
                        "service_type": service_type,
                        "plan_or_loan_name": autopay_case.plan_or_loan_name if autopay_case else "your plan",
                        "due_amount": float(autopay_case.due_amount or 0.0) if autopay_case else 0.0,
                        "currency": autopay_case.currency if autopay_case else "INR",
                        "current_step": "greeting", "current_language": "English",
                        "is_identity_confirmed": None, "detected_intent": "none",
                        "silence_count": 0
                    }
                    await _tts_and_stream(fallback_text, "English")

            elif event == "media":
                now_ts = time.monotonic()
                elapsed = now_ts - last_media_ts
                last_media_ts = now_ts

                if not session_state or is_playing_audio:
                    continue
                if time.monotonic() < vad_muted_until:
                    continue

                payload = data.get("media", {}).get("payload", "")
                if not payload:
                    continue
                try:
                    chunk = base64.b64decode(payload)
                except Exception:
                    continue

                rms = _get_rms(chunk)
                if not is_speaking:
                    if rms > SPEECH_THRESHOLD:
                        is_speaking = True
                        utterance_start_ts = time.monotonic()
                        audio_buffer.extend(chunk)
                        silence_dur = 0.0
                        no_speech_dur = 0.0
                    else:
                        no_speech_dur += elapsed
                        if no_speech_dur >= 6.0:
                            no_speech_dur = 0.0
                            call_completed = await _run_agent_turn("")
                            if call_completed:
                                await asyncio.sleep(2.0)
                                break
                else:
                    audio_buffer.extend(chunk)
                    utterance_dur = time.monotonic() - utterance_start_ts
                    if rms < SILENCE_THRESHOLD:
                        silence_dur += elapsed
                        force_stt = (silence_dur >= SILENCE_TIMEOUT_MS / 1000.0)
                    else:
                        silence_dur = 0.0
                        force_stt = None

                    if force_stt is None and utterance_dur >= MAX_UTTERANCE_DURATION_S:
                        force_stt = True

                    if force_stt is True:
                        raw_pcm = bytes(audio_buffer)
                        audio_buffer.clear()
                        is_speaking = False
                        silence_dur = 0.0
                        no_speech_dur = 0.0

                        temp_wav = f"static/tmp_{call_sid}.wav"
                        _save_pcm_to_wav(raw_pcm, temp_wav)
                        try:
                            with open(temp_wav, "rb") as f:
                                wav_bytes = f.read()
                            user_text = await loop.run_in_executor(
                                None, STTService.speech_to_text, wav_bytes,
                                session_state.get("current_language", "English")
                            )
                        except Exception as exc:
                            logger.error(f"[STT] ERROR | call_sid={call_sid} | error={exc}")
                            user_text = ""
                        finally:
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)

                        call_completed = await _run_agent_turn(user_text)
                        if call_completed:
                            await asyncio.sleep(2.0)
                            break

            elif event == "stop":
                logger.info(f"[WS] STOP EVENT | call_sid={call_sid}")
                break

    except WebSocketDisconnect:
        logger.info(f"[WS] DISCONNECTED | call_sid={call_sid}")
    except Exception as exc:
        logger.error(f"[WS] Unexpected error: {exc} | call_sid={call_sid}", exc_info=True)
    finally:
        if session_state and session_state.get("attempt_id"):
            att_id = session_state.get("attempt_id")
            try:
                from database.models import CallAttempt as CA
                attempt_obj = db.query(CA).filter(CA.id == att_id).first()
                if attempt_obj and (attempt_obj.call_outcome is None or attempt_obj.call_status in ["INITIATED", "CONNECTED"]):
                    last_step = session_state.get("current_step") or "unknown"
                    final_disp = session_state.get("final_disposition") or "Disconnected"
                    attempt_obj.call_status = "DISCONNECTED"
                    attempt_obj.call_outcome = final_disp
                    attempt_obj.end_time = datetime.utcnow()
                    if attempt_obj.start_time:
                        dur_s = (attempt_obj.end_time - attempt_obj.start_time).total_seconds()
                        attempt_obj.minutes_spoken = round(dur_s / 60.0, 2)
                    attempt_obj.disconnected_by = "Customer"

                    conv_hist = []
                    for m in session_state.get("messages", []):
                        role = "bot" if isinstance(m, AIMessage) else "user"
                        conv_hist.append({"role": role, "text": m.content})
                    attempt_obj.conversation_history = conv_hist
                    db.commit()
            except Exception as e:
                logger.error(f"[WS] Error saving disconnected attempt: {e}")
        db.close()


# ==========================================
# SIMULATION / CHAT PLAYGROUND API
# ==========================================
@app.post("/api/simulate/chat")
async def simulate_chat_turn(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Simulation endpoint allowing real-time testing of multi-turn conversations
    with the Autopay Recovery Voice Agent without requiring telephony.
    """
    data = await request.json()
    case_id = data.get("case_id", 1)
    user_message = data.get("message", "").strip()
    session_history = data.get("history", [])  # List of {role: 'user'/'bot', content: '...'}
    current_state = data.get("state")

    autopay_case = crud.get_autopay_case(db=db, case_id=case_id)
    if not autopay_case:
        # Fallback to first available case
        cases = crud.get_all_autopay_cases(db=db)
        if cases:
            autopay_case = cases[0]
            case_id = autopay_case.id
        else:
            raise HTTPException(status_code=404, detail="No customer records found. Please seed data first.")

    # Reconstruct messages
    messages = []
    for item in session_history:
        if item.get("role") in ["user", "human"]:
            messages.append(HumanMessage(content=item.get("content", "")))
        else:
            messages.append(AIMessage(content=item.get("content", "")))

    if user_message:
        messages.append(HumanMessage(content=user_message))

    # Construct AgentState
    state: AgentState = current_state or {
        "messages": messages,
        "case_id": autopay_case.id,
        "attempt_id": 0,
        "customer_id": autopay_case.customer_id or "CUST-1001",
        "customer_name": autopay_case.customer_name or "Customer",
        "mobile": autopay_case.mobile or "",
        "email": autopay_case.email or "",
        "service_type": autopay_case.service_type or "Subscription",
        "plan_or_loan_name": autopay_case.plan_or_loan_name or "Plan",
        "due_amount": float(autopay_case.due_amount or 0.0),
        "currency": autopay_case.currency or "INR",
        "due_date": autopay_case.due_date.strftime("%Y-%m-%d") if autopay_case.due_date else "",
        "failed_date": autopay_case.failed_date.strftime("%Y-%m-%d") if autopay_case.failed_date else "",
        "payment_method": autopay_case.payment_method or "Autopay",
        "failure_reason": autopay_case.failure_reason or "Insufficient Balance",
        "payment_link": autopay_case.payment_link or "https://rzp.io/pay",
        "is_callback_retry": False,
        "current_step": "greeting" if not messages else (current_state.get("current_step") if current_state else "reason_discovery"),
        "current_language": current_state.get("current_language", "English") if current_state else "English",
        "is_identity_confirmed": current_state.get("is_identity_confirmed") if current_state else None,
        "detected_intent": "none",
        "previous_intent": None,
        "intent_changed": False,
        "intent_confidence": 1.0,
        "is_off_topic": False,
        "agreed_retry_date": None,
        "callback_time": None,
        "dispute_reason": None,
        "final_disposition": None,
        "customer_sentiment": "Neutral",
        "call_summary": None,
        "payment_link_sent": False,
        "silence_count": 0,
        "clarification_count": 0,
        "is_silence_turn": False
    }

    state["messages"] = messages
    next_state = autopay_agent.invoke(state)

    out_msgs = next_state.get("messages", [])
    bot_reply = out_msgs[-1].content if out_msgs else "Thank you!"

    # Serialize history
    updated_history = []
    for m in out_msgs:
        role = "bot" if isinstance(m, AIMessage) else "user"
        updated_history.append({"role": role, "content": m.content})

    return {
        "reply": bot_reply,
        "step": next_state.get("current_step"),
        "intent": next_state.get("detected_intent"),
        "sentiment": next_state.get("customer_sentiment"),
        "final_disposition": next_state.get("final_disposition"),
        "agreed_retry_date": next_state.get("agreed_retry_date"),
        "callback_time": next_state.get("callback_time"),
        "payment_link_sent": next_state.get("payment_link_sent"),
        "history": updated_history,
        "is_completed": (next_state.get("current_step") == "completed"),
        "state": {k: v for k, v in next_state.items() if k != "messages"}
    }


# ==========================================
# REST API: CUSTOMER RECORDS & CAMPAIGNS
# ==========================================
@app.get("/api/cases")
def list_autopay_cases(
    campaign_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all Autopay recovery cases with call attempt metrics."""
    query = db.query(AutopayCase)
    if campaign_id:
        query = query.filter(AutopayCase.campaign_id == campaign_id)
    if status:
        query = query.filter(AutopayCase.agent_status == status)

    cases = query.order_by(AutopayCase.id.asc()).all()
    results = []
    for c in cases:
        attempts = crud.get_call_attempts_for_case(db=db, case_id=c.id)
        results.append({
            "id": c.id,
            "campaign_id": c.campaign_id,
            "customer_id": c.customer_id,
            "customer_name": c.customer_name,
            "mobile": c.mobile,
            "email": c.email,
            "service_type": c.service_type,
            "plan_or_loan_name": c.plan_or_loan_name,
            "due_amount": float(c.due_amount or 0.0),
            "currency": c.currency,
            "due_date": c.due_date.strftime("%Y-%m-%d") if c.due_date else None,
            "failed_date": c.failed_date.strftime("%Y-%m-%d") if c.failed_date else None,
            "payment_method": c.payment_method,
            "failure_reason": c.failure_reason,
            "payment_link": c.payment_link,
            "agent_status": c.agent_status,
            "final_disposition": c.final_disposition,
            "customer_sentiment": c.customer_sentiment,
            "agreed_retry_date": c.agreed_retry_date.strftime("%Y-%m-%d") if c.agreed_retry_date else None,
            "callback_scheduled_at": c.callback_scheduled_at.strftime("%Y-%m-%d %H:%M") if c.callback_scheduled_at else None,
            "payment_status": c.payment_status,
            "call_summary": c.call_summary,
            "total_attempts": len(attempts),
            "created_at": c.created_at.strftime("%Y-%m-%d %H:%M") if c.created_at else None
        })
    return {"total": len(results), "cases": results}


@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: int, db: Session = Depends(get_db)):
    case = crud.get_autopay_case(db=db, case_id=case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    attempts = crud.get_call_attempts_for_case(db=db, case_id=case_id)
    attempts_data = []
    for att in attempts:
        attempts_data.append({
            "attempt_number": att.attempt_number,
            "call_status": att.call_status,
            "call_outcome": att.call_outcome,
            "minutes_spoken": float(att.minutes_spoken or 0.0),
            "recording_url": att.recording_url,
            "feedback": att.feedback,
            "conversation_history": att.conversation_history,
            "created_at": att.created_at.strftime("%Y-%m-%d %H:%M:%S") if att.created_at else None
        })

    return {
        "case": {
            "id": case.id,
            "customer_id": case.customer_id,
            "customer_name": case.customer_name,
            "mobile": case.mobile,
            "email": case.email,
            "service_type": case.service_type,
            "plan_or_loan_name": case.plan_or_loan_name,
            "due_amount": float(case.due_amount or 0.0),
            "currency": case.currency,
            "due_date": case.due_date.strftime("%Y-%m-%d") if case.due_date else None,
            "failed_date": case.failed_date.strftime("%Y-%m-%d") if case.failed_date else None,
            "payment_method": case.payment_method,
            "failure_reason": case.failure_reason,
            "payment_link": case.payment_link,
            "agent_status": case.agent_status,
            "final_disposition": case.final_disposition,
            "customer_sentiment": case.customer_sentiment,
            "agreed_retry_date": case.agreed_retry_date.strftime("%Y-%m-%d") if case.agreed_retry_date else None,
            "callback_scheduled_at": case.callback_scheduled_at.strftime("%Y-%m-%d %H:%M") if case.callback_scheduled_at else None,
            "payment_status": case.payment_status,
            "call_summary": case.call_summary
        },
        "attempts": attempts_data
    }


@app.post("/api/seed-data")
def seed_dummy_records(db: Session = Depends(get_db)):
    """Seed the 10 fictional customer records into the DB."""
    records_file = "dummy_autopay_records.xlsx" if os.path.exists("dummy_autopay_records.xlsx") else "dummy_autopay_records.csv"
    if not os.path.exists(records_file):
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    parsed_cases = parse_autopay_file(records_file)
    campaign = crud.create_campaign(
        db=db,
        name="10 Fictional Autopay Recovery Campaign",
        file_name=records_file,
        total_records=len(parsed_cases)
    )

    for case_data in parsed_cases:
        crud.create_autopay_case(db=db, campaign_id=campaign.id, data=case_data)

    campaign.status = "READY"
    db.commit()

    return {
        "status": "success",
        "message": f"Successfully created campaign #{campaign.id} with {len(parsed_cases)} fictional records!",
        "campaign_id": campaign.id,
        "total_records": len(parsed_cases)
    }


@app.post("/api/campaigns/upload", status_code=status.HTTP_201_CREATED)
async def upload_campaign_file(
    name: str = Form(..., description="Campaign Name"),
    file: UploadFile = File(..., description="Excel (.xlsx) or CSV file with autopay records"),
    db: Session = Depends(get_db)
):
    """Uploads an Excel or CSV file containing autopay records and creates a campaign."""
    try:
        contents = await file.read()
        parsed_cases = parse_autopay_file(contents, filename=file.filename)
        if not parsed_cases:
            raise HTTPException(status_code=400, detail="No valid customer records found in the uploaded file.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")

    campaign = crud.create_campaign(
        db=db,
        name=name,
        file_name=file.filename,
        total_records=len(parsed_cases)
    )

    for case_data in parsed_cases:
        crud.create_autopay_case(db=db, campaign_id=campaign.id, data=case_data)

    campaign.status = "READY"
    db.commit()

    return {
        "campaign_id": campaign.id,
        "campaign_name": campaign.name,
        "file_name": campaign.file_name,
        "total_records_inserted": len(parsed_cases),
        "status": campaign.status
    }


@app.get("/api/campaigns")
def list_campaigns(db: Session = Depends(get_db)):
    campaigns = crud.get_all_campaigns(db=db)
    return [{
        "id": camp.id,
        "name": camp.name,
        "file_name": camp.file_name,
        "total_records": camp.total_records,
        "status": camp.status,
        "created_at": camp.created_at.strftime("%Y-%m-%d %H:%M:%S") if camp.created_at else None
    } for camp in campaigns]


# ==========================================
# OUTBOUND CALL TRIGGERS & WEBHOOKS
# ==========================================
def _prepare_greeting_and_initiate_call(autopay_case, webhook_domain: str, db: Session) -> dict:
    global _global_webhook_domain
    _global_webhook_domain = webhook_domain.rstrip("/") if webhook_domain else _global_webhook_domain

    attempts = crud.get_call_attempts_for_case(db=db, case_id=autopay_case.id)
    attempt = crud.create_call_attempt(
        db=db,
        case_id=autopay_case.id,
        attempt_number=len(attempts) + 1,
        provider_call_id=f"pre_{autopay_case.id}"
    )
    crud.update_autopay_case_status(db=db, case_id=autopay_case.id, agent_status="CALLING")

    initial_state: AgentState = {
        "messages": [],
        "case_id": autopay_case.id,
        "attempt_id": attempt.id,
        "customer_id": autopay_case.customer_id or "CUST-1001",
        "customer_name": autopay_case.customer_name or "Customer",
        "mobile": autopay_case.mobile or "",
        "email": autopay_case.email or "",
        "service_type": autopay_case.service_type or "Subscription",
        "plan_or_loan_name": autopay_case.plan_or_loan_name or "Plan",
        "due_amount": float(autopay_case.due_amount or 0.0),
        "currency": autopay_case.currency or "INR",
        "due_date": autopay_case.due_date.strftime("%Y-%m-%d") if autopay_case.due_date else "",
        "failed_date": autopay_case.failed_date.strftime("%Y-%m-%d") if autopay_case.failed_date else "",
        "payment_method": autopay_case.payment_method or "Autopay",
        "failure_reason": autopay_case.failure_reason or "Insufficient Balance",
        "payment_link": autopay_case.payment_link or "https://rzp.io/pay",
        "is_callback_retry": (autopay_case.final_disposition == "Callback Requested"),
        "current_step": "greeting",
        "current_language": "English",
        "is_identity_confirmed": None,
        "detected_intent": "none",
        "previous_intent": None,
        "intent_changed": False,
        "intent_confidence": 1.0,
        "is_off_topic": False,
        "agreed_retry_date": None,
        "callback_time": None,
        "dispute_reason": None,
        "final_disposition": None,
        "customer_sentiment": "Neutral",
        "call_summary": None,
        "payment_link_sent": False,
        "silence_count": 0,
        "clarification_count": 0,
        "is_silence_turn": False
    }

    agent_output = autopay_agent.invoke(initial_state)
    messages = agent_output.get("messages", [])
    greeting_text = messages[-1].content if messages else f"Hello, this is Arya calling from Razorpay regarding your {autopay_case.service_type} payment."

    audio_filename = TTSService.text_to_speech(
        greeting_text,
        language=agent_output.get("current_language", "English")
    )
    wav_local_path = os.path.join("static", audio_filename)

    call_result = ExotelService.initiate_call(
        customer_mobile=autopay_case.mobile,
        case_id=autopay_case.id,
        webhook_domain=webhook_domain
    )

    call_sid_from_exotel = (
        call_result.get("Call", {}).get("Sid")
        or call_result.get("call_sid")
        or call_result.get("sid")
        or ""
    )

    if call_sid_from_exotel:
        attempt.provider_call_id = call_sid_from_exotel
        db.commit()

    host_domain = (webhook_domain or _global_webhook_domain or "").rstrip("/")
    audio_public_url = f"{host_domain}/audio/{audio_filename}" if host_domain else f"/audio/{audio_filename}"

    greeting_data = {
        "wav_file": wav_local_path,
        "audio_url": audio_public_url,
        "agent_state": agent_output,
        "case_id": str(autopay_case.id),
        "call_sid": call_sid_from_exotel
    }

    if call_sid_from_exotel:
        pending_greetings[call_sid_from_exotel] = greeting_data
    pending_greetings[str(autopay_case.id)] = greeting_data

    return call_result


@app.post("/calls/initiate/{case_id}")
def initiate_call_endpoint(
    case_id: int,
    webhook_domain: Optional[str] = Query(None, description="Public server domain (e.g. https://your-ngrok.io)"),
    db: Session = Depends(get_db)
):
    case = crud.get_autopay_case(db=db, case_id=case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if not webhook_domain:
        raise HTTPException(status_code=400, detail="webhook_domain is required")

    try:
        res = _prepare_greeting_and_initiate_call(case, webhook_domain, db)
        return {"status": "success", "case_id": case_id, "exotel_response": res}
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/campaigns/{campaign_id}/start")
def start_campaign_calling_endpoint(
    campaign_id: int,
    webhook_domain: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    campaign = crud.get_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    pending_cases = db.query(AutopayCase).filter(
        AutopayCase.campaign_id == campaign_id,
        AutopayCase.agent_status == "PENDING"
    ).all()

    triggered = []
    failed = []

    for case in pending_cases:
        try:
            _prepare_greeting_and_initiate_call(case, webhook_domain, db)
            triggered.append(case.id)
        except Exception as e:
            failed.append({"case_id": case.id, "error": str(e)})

    campaign.status = "PROCESSING"
    db.commit()

    return {
        "campaign_id": campaign_id,
        "total_pending": len(pending_cases),
        "triggered_count": len(triggered),
        "triggered_case_ids": triggered,
        "failed": failed
    }


# ==========================================
# EXCEL REPORT EXPORT
# ==========================================
@app.get("/campaigns/{campaign_id}/report")
def export_campaign_report(
    campaign_id: int,
    db: Session = Depends(get_db)
):
    campaign = crud.get_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    cases = db.query(AutopayCase).filter(AutopayCase.campaign_id == campaign_id).all()
    rows = []
    for c in cases:
        attempts = crud.get_call_attempts_for_case(db=db, case_id=c.id)
        rows.append({
            "Case ID": c.id,
            "Customer ID": c.customer_id,
            "Customer Name": c.customer_name,
            "Mobile": c.mobile,
            "Service Type": c.service_type,
            "Plan / Loan Name": c.plan_or_loan_name,
            "Due Amount (INR)": float(c.due_amount or 0.0),
            "Due Date": c.due_date.strftime("%Y-%m-%d") if c.due_date else "",
            "Payment Method": c.payment_method,
            "Failure Reason": c.failure_reason,
            "Payment Status": c.payment_status,
            "Agent Status": c.agent_status,
            "Final Disposition": c.final_disposition or "-",
            "Agreed Retry Date": c.agreed_retry_date.strftime("%Y-%m-%d") if c.agreed_retry_date else "-",
            "Callback Time": c.callback_scheduled_at.strftime("%Y-%m-%d %H:%M") if c.callback_scheduled_at else "-",
            "Total Call Attempts": len(attempts),
            "Call Summary": c.call_summary or ""
        })

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Autopay Recovery Report")

    output.seek(0)
    filename = f"autopay_campaign_{campaign_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return StreamingResponse(
        output,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ==========================================
# INTERACTIVE WEB UI DASHBOARD
# ==========================================
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard_html():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Razorpay Autopay Recovery Voice Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-card: #111827;
            --bg-card-hover: #1f293d;
            --accent-blue: #2563eb;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.08);
            --glass-bg: rgba(17, 24, 39, 0.85);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 24px; }
        
        .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 24px; border-bottom: 1px solid var(--border-color); margin-bottom: 28px; }
        .logo-group { display: flex; align-items: center; gap: 14px; }
        .logo-icon { width: 48px; height: 48px; background: linear-gradient(135deg, #0284c7, #2563eb, #7c3aed); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; color: #fff; box-shadow: 0 8px 20px rgba(37, 99, 235, 0.35); }
        .title-text h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; background: linear-gradient(90deg, #fff, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .title-text p { font-size: 13px; color: var(--text-muted); }

        .actions-group { display: flex; gap: 12px; align-items: center; }
        .btn { padding: 10px 18px; border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer; border: none; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s; }
        .btn-primary { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35); }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5); }
        .btn-secondary { background: rgba(255, 255, 255, 0.06); color: var(--text-main); border: 1px solid var(--border-color); }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
        .btn-success { background: linear-gradient(135deg, #059669, #10b981); color: white; }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 28px; }
        .stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 20px; position: relative; overflow: hidden; }
        .stat-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #2563eb, #06b6d4); }
        .stat-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 8px; }
        .stat-val { font-size: 28px; font-weight: 800; color: #fff; }
        .stat-sub { font-size: 12px; color: var(--accent-green); margin-top: 4px; display: flex; align-items: center; gap: 4px; }

        .main-layout { display: grid; grid-template-columns: 1fr 380px; gap: 24px; }
        @media(max-width: 1080px) { .main-layout { grid-template-columns: 1fr; } }

        .panel { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 22px; margin-bottom: 24px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
        .panel-title { font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 10px; color: #fff; }

        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th { padding: 12px 14px; background: rgba(255, 255, 255, 0.03); color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border-color); }
        td { padding: 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }

        .badge { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
        .badge-pending { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-calling { background: rgba(37, 99, 235, 0.15); color: #60a5fa; border: 1px solid rgba(37, 99, 235, 0.3); }
        .badge-completed { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-failed { background: rgba(244, 63, 94, 0.15); color: #f87171; border: 1px solid rgba(244, 63, 94, 0.3); }

        /* Simulator Styles */
        .chat-box { display: flex; flex-direction: column; height: 420px; background: rgba(0, 0, 0, 0.25); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
        .chat-messages { flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
        .msg { max-width: 84%; padding: 12px 16px; border-radius: 12px; font-size: 13px; line-height: 1.5; }
        .msg-bot { background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(6, 182, 212, 0.15)); border: 1px solid rgba(37, 99, 235, 0.4); align-self: flex-start; border-bottom-left-radius: 2px; }
        .msg-user { background: #1e293b; border: 1px solid var(--border-color); align-self: flex-end; border-bottom-right-radius: 2px; }
        .chat-input-row { display: flex; padding: 12px; background: rgba(255, 255, 255, 0.02); border-top: 1px solid var(--border-color); gap: 8px; }
        .chat-input-row input { flex: 1; background: #0f172a; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 14px; color: #fff; font-size: 13px; outline: none; }
        .chat-input-row input:focus { border-color: var(--accent-blue); }

        .quick-chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 12px; background: rgba(0, 0, 0, 0.3); border-top: 1px solid var(--border-color); }
        .chip { background: rgba(255, 255, 255, 0.06); border: 1px solid var(--border-color); border-radius: 20px; font-size: 11px; padding: 4px 10px; cursor: pointer; color: var(--text-muted); }
        .chip:hover { background: rgba(37, 99, 235, 0.2); color: #93c5fd; border-color: #2563eb; }
    </style>
</head>
<body>

    <div class="header">
        <div class="logo-group">
            <div class="logo-icon"><i class="fa-solid fa-phone-volume"></i></div>
            <div class="title-text">
                <h1>Razorpay Autopay Recovery Voice Agent</h1>
                <p>AI-Powered Automated Failed Autopay Resolution & Telephony Console</p>
            </div>
        </div>
        <div class="actions-group">
            <button class="btn btn-secondary" onclick="seedData()"><i class="fa-solid fa-database"></i> Seed 10 Fictional Records</button>
            <button class="btn btn-primary" onclick="loadCases()"><i class="fa-solid fa-arrows-rotate"></i> Refresh Data</button>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-title">Total Records</div>
            <div class="stat-val" id="stat-total">0</div>
            <div class="stat-sub"><i class="fa-solid fa-users"></i> Across Campaigns</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Total Recoverable Due</div>
            <div class="stat-val" id="stat-amount">₹0</div>
            <div class="stat-sub" style="color: #60a5fa;"><i class="fa-solid fa-indian-rupee-sign"></i> Failed Autopay Value</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Pending Calls</div>
            <div class="stat-val" id="stat-pending" style="color: #f59e0b;">0</div>
            <div class="stat-sub" style="color: #f59e0b;"><i class="fa-solid fa-clock"></i> Ready for Outreach</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Resolved / Concluded</div>
            <div class="stat-val" id="stat-completed" style="color: #10b981;">0</div>
            <div class="stat-sub"><i class="fa-solid fa-check-circle"></i> Completed Calls</div>
        </div>
    </div>

    <div class="main-layout">
        <!-- Main Customer Records Table -->
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><i class="fa-solid fa-list-check" style="color: #60a5fa;"></i> Fictional Customer Autopay Records</div>
                <span id="record-count-badge" class="badge badge-calling">10 Records</span>
            </div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Customer</th>
                            <th>Service / Loan</th>
                            <th>Failed Amount</th>
                            <th>Failure Reason</th>
                            <th>Payment Method</th>
                            <th>Status</th>
                            <th>Disposition</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="cases-table-body">
                        <tr><td colspan="8" style="text-align:center; padding: 30px; color: var(--text-muted);">Loading customer records...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Live Simulation Console -->
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title"><i class="fa-solid fa-robot" style="color: #06b6d4;"></i> Voice Agent Simulator</div>
                <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="resetSimulator()"><i class="fa-solid fa-rotate-left"></i> Reset</button>
            </div>
            
            <div style="margin-bottom: 12px;">
                <label style="font-size: 12px; color: var(--text-muted);">Test Against Customer Record:</label>
                <select id="sim-customer-select" style="width: 100%; margin-top: 4px; padding: 8px; background: #0f172a; border: 1px solid var(--border-color); color: #fff; border-radius: 6px; font-size: 13px;" onchange="resetSimulator()">
                    <option value="1">CUST-1001: Aarav Sharma (₹2,499 - SaaS Subscription)</option>
                </select>
            </div>

            <div class="chat-box">
                <div class="chat-messages" id="sim-chat-msgs">
                    <div class="msg msg-bot">
                        <strong>Arya (Agent):</strong><br>
                        Hello, good day! This is Arya calling from Razorpay regarding your SaaS Subscription payment. Am I speaking with Aarav Sharma?
                    </div>
                </div>

                <div class="quick-chips">
                    <span class="chip" onclick="sendQuick('Yes, speaking')">Yes, speaking</span>
                    <span class="chip" onclick="sendQuick('Send me payment link')">Send payment link</span>
                    <span class="chip" onclick="sendQuick('Retry tomorrow, will add balance')">Retry tomorrow</span>
                    <span class="chip" onclick="sendQuick('My card is expired')">Card expired</span>
                    <span class="chip" onclick="sendQuick('Call me back at 5 PM')">Callback at 5 PM</span>
                    <span class="chip" onclick="sendQuick('I already paid this yesterday')">Already paid</span>
                </div>

                <div class="chat-input-row">
                    <input type="text" id="sim-input" placeholder="Type customer reply..." onkeypress="handleSimEnter(event)">
                    <button class="btn btn-primary" style="padding: 8px 14px;" onclick="sendSimMessage()"><i class="fa-solid fa-paper-plane"></i></button>
                </div>
            </div>

            <div id="sim-meta" style="margin-top: 12px; font-size: 11px; color: var(--text-muted); background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                Step: <strong>Greeting</strong> | Intent: <strong>None</strong> | Sentiment: <strong>Neutral</strong>
            </div>
        </div>
    </div>

    <script>
        let simHistory = [
            { role: "bot", content: "Hello, good day! This is Arya calling from Razorpay regarding your SaaS Subscription payment. Am I speaking with Aarav Sharma?" }
        ];
        let simState = null;
        let allCases = [];

        async function loadCases() {
            try {
                const res = await fetch('/api/cases');
                const data = await res.json();
                allCases = data.cases || [];

                let totalAmount = 0;
                let pendingCount = 0;
                let completedCount = 0;

                const tbody = document.getElementById('cases-table-body');
                const select = document.getElementById('sim-customer-select');
                
                if (allCases.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding: 30px; color: var(--text-muted);">No records found. Click "Seed 10 Fictional Records" above to load data.</td></tr>';
                    return;
                }

                tbody.innerHTML = '';
                select.innerHTML = '';

                allCases.forEach((c, idx) => {
                    totalAmount += c.due_amount;
                    if (c.agent_status === 'PENDING') pendingCount++;
                    if (c.agent_status === 'COMPLETED') completedCount++;

                    let statusBadge = '<span class="badge badge-pending">Pending</span>';
                    if (c.agent_status === 'CALLING') statusBadge = '<span class="badge badge-calling">Calling</span>';
                    if (c.agent_status === 'COMPLETED') statusBadge = '<span class="badge badge-completed">Completed</span>';
                    if (c.agent_status === 'FAILED') statusBadge = '<span class="badge badge-failed">Failed</span>';

                    const row = `
                        <tr>
                            <td>
                                <strong style="color: #fff;">${c.customer_name}</strong><br>
                                <span style="font-size: 11px; color: var(--text-muted);">${c.customer_id} • ${c.mobile}</span>
                            </td>
                            <td>
                                <span>${c.service_type}</span><br>
                                <span style="font-size: 11px; color: var(--text-muted);">${c.plan_or_loan_name}</span>
                            </td>
                            <td><strong style="color: #34d399;">₹${c.due_amount.toLocaleString()}</strong></td>
                            <td style="font-size: 12px; color: #f87171;">${c.failure_reason || '-'}</td>
                            <td style="font-size: 12px;">${c.payment_method || '-'}</td>
                            <td>${statusBadge}</td>
                            <td style="font-size: 12px; color: #93c5fd;">${c.final_disposition || '-'}</td>
                            <td>
                                <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="testSimForCase(${c.id})"><i class="fa-solid fa-play"></i> Simulate</button>
                            </td>
                        </tr>
                    `;
                    tbody.innerHTML += row;

                    select.innerHTML += `<option value="${c.id}">${c.customer_id}: ${c.customer_name} (₹${c.due_amount} - ${c.service_type})</option>`;
                });

                document.getElementById('stat-total').innerText = allCases.length;
                document.getElementById('stat-amount').innerText = '₹' + totalAmount.toLocaleString();
                document.getElementById('stat-pending').innerText = pendingCount;
                document.getElementById('stat-completed').innerText = completedCount;
                document.getElementById('record-count-badge').innerText = allCases.length + ' Records';

            } catch (err) {
                console.error("Error loading cases:", err);
            }
        }

        async function seedData() {
            try {
                const res = await fetch('/api/seed-data', { method: 'POST' });
                const data = await res.json();
                alert(data.message || "Data seeded successfully!");
                loadCases();
            } catch (err) {
                alert("Failed to seed data: " + err);
            }
        }

        function testSimForCase(caseId) {
            document.getElementById('sim-customer-select').value = caseId;
            resetSimulator();
        }

        function resetSimulator() {
            const caseId = parseInt(document.getElementById('sim-customer-select').value) || 1;
            const targetCase = allCases.find(c => c.id === caseId) || { customer_name: "Customer", service_type: "Subscription" };

            simHistory = [
                { role: "bot", content: `Hello, good day! This is Arya calling from Razorpay regarding your ${targetCase.service_type} payment. Am I speaking with ${targetCase.customer_name}?` }
            ];
            simState = null;

            const chat = document.getElementById('sim-chat-msgs');
            chat.innerHTML = `
                <div class="msg msg-bot">
                    <strong>Arya (Agent):</strong><br>
                    ${simHistory[0].content}
                </div>
            `;
            document.getElementById('sim-meta').innerHTML = 'Step: <strong>Greeting</strong> | Intent: <strong>None</strong> | Sentiment: <strong>Neutral</strong>';
        }

        function handleSimEnter(e) {
            if (e.key === 'Enter') sendSimMessage();
        }

        function sendQuick(text) {
            document.getElementById('sim-input').value = text;
            sendSimMessage();
        }

        async function sendSimMessage() {
            const input = document.getElementById('sim-input');
            const text = input.value.trim();
            if (!text) return;
            input.value = '';

            const chat = document.getElementById('sim-chat-msgs');
            chat.innerHTML += `<div class="msg msg-user"><strong>Customer:</strong><br>${text}</div>`;
            chat.scrollTop = chat.scrollHeight;

            const caseId = parseInt(document.getElementById('sim-customer-select').value) || 1;

            try {
                const res = await fetch('/api/simulate/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        case_id: caseId,
                        message: text,
                        history: simHistory,
                        state: simState
                    })
                });
                const data = await res.json();

                chat.innerHTML += `<div class="msg msg-bot"><strong>Arya (Agent):</strong><br>${data.reply}</div>`;
                chat.scrollTop = chat.scrollHeight;

                simHistory = data.history;
                simState = data.state;

                document.getElementById('sim-meta').innerHTML = `Step: <strong>${data.step}</strong> | Intent: <strong>${data.intent}</strong> | Sentiment: <strong>${data.sentiment}</strong> | Disp: <strong>${data.final_disposition || 'Pending'}</strong>`;

                if (data.is_completed) {
                    loadCases();
                }

            } catch (err) {
                console.error("Simulation error:", err);
            }
        }

        // On Load
        loadCases();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ==========================================
# APP LIFECYCLE EVENTS
# ==========================================
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
    validate_sarvam_config()


@app.on_event("shutdown")
def shutdown_event():
    close_http_client()
    logger.info("Sarvam HTTP client closed.")
