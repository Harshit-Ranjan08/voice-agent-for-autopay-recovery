import os
import logging
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("services.exotel")

# ==========================================
# ENVIRONMENT CONFIG
# ==========================================
EXOTEL_API_KEY      = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN    = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_ACCOUNT_SID  = os.getenv("EXOTEL_ACCOUNT_SID")
EXOTEL_EXOPHONE     = os.getenv("EXOTEL_EXOPHONE")
EXOTEL_APP_ID       = os.getenv("EXOTEL_APP_ID")
EXOTEL_API_HOST     = os.getenv("EXOTEL_API_HOST", "api.exotel.com")

# ==========================================
# HYPERPARAMETERS (Best for NDR Calling)
# ==========================================

# TimeLimit: max call duration in seconds (10 minutes = 600 seconds)
# Prevents runaway costs if the agent gets stuck or the customer doesn't disconnect
CALL_TIME_LIMIT_SECONDS = 200

# TimeOut: ring timeout before giving up (30 seconds)
# If the customer doesn't pick up in 30s, Exotel marks it as No Answer
CALL_RING_TIMEOUT_SECONDS = 30

# CallType: "trans" = transactional call
# Transactional calls are NOT restricted by DND (Do Not Disturb) registry rules
CALL_TYPE = "trans"

# Record: enable call recording in Exotel
CALL_RECORD = True

# StatusCallbackEvents: send callbacks on both "answered" AND "terminal" (call ended)
# "answered" → updates DB when customer picks up
# "terminal" → updates DB with final call status (Completed / No Answer / Busy / Failed)
STATUS_CALLBACK_EVENTS = ["answered", "terminal"]

# StatusCallbackContentType: receive callback data as JSON (easier to parse in FastAPI)
STATUS_CALLBACK_CONTENT_TYPE = "application/json"


def _validate_credentials():
    """Validates that all required Exotel credentials are available."""
    missing = []
    if not EXOTEL_API_KEY:
        missing.append("EXOTEL_API_KEY")
    if not EXOTEL_API_TOKEN:
        missing.append("EXOTEL_API_TOKEN")
    if not EXOTEL_ACCOUNT_SID:
        missing.append("EXOTEL_ACCOUNT_SID")
    if not EXOTEL_EXOPHONE:
        missing.append("EXOTEL_EXOPHONE")
    if not EXOTEL_APP_ID:
        missing.append("EXOTEL_APP_ID")
    if missing:
        raise ValueError(f"Missing Exotel environment variables: {', '.join(missing)}")


def _build_call_url() -> str:
    """Builds the Exotel API endpoint URL for connecting calls."""
    return f"https://{EXOTEL_API_HOST}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls/connect.json"


def _build_flow_url() -> str:
    """
    Builds the ExoML voice flow URL that Exotel fetches when the customer answers.
    This flow is defined inside the Exotel dashboard (App Bazaar).
    """
    return f"http://my.exotel.com/{EXOTEL_ACCOUNT_SID}/exoml/start_voice/{EXOTEL_APP_ID}"


# ==========================================
# EXOTEL SERVICE
# ==========================================

class ExotelService:

    @staticmethod
    def initiate_call(
        customer_mobile: str,
        case_id: int,
        webhook_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initiates an outbound call via Exotel to a customer phone number.

        Exotel will:
        1. Dial the customer's mobile number (From)
        2. When the customer picks up, fetch the ExoML flow URL (Url)
        3. The flow instructs Exotel to <Play> the greeting audio and <Record> the response
        4. At the end of the call, Exotel hits the StatusCallback URL with final call details

        Args:
            customer_mobile (str): Customer's 10-digit mobile number (e.g., "8303464646")
            case_id (int): The DB ID of the NDR case, passed as CustomField for webhook correlation
            webhook_domain (str, optional): Your public server domain (e.g., "https://your-ngrok-url.io")
                                           Used to receive Exotel status callbacks.

        Returns:
            dict: Parsed JSON response from Exotel API containing call details:
                  { "Call": { "Sid": "...", "Status": "queued", "From": "...", ... } }

        Raises:
            ValueError: If any required Exotel credentials are missing in the environment
            Exception: If the Exotel API returns a non-success status code
        """
        _validate_credentials()

        # Normalize phone number: ensure it starts without country code prefix issues
        # Exotel expects numbers in local format (e.g., 08303464646 or 8303464646)
        mobile = customer_mobile.strip()
        if not mobile.startswith("0") and len(mobile) == 10:
            mobile = "0" + mobile  # Prepend 0 for domestic format required by Exotel

        api_url = _build_call_url()
        # Always use Exotel App Bazaar flow URL as required by Exotel connect.json API
        flow_url = _build_flow_url()


        # Build payload
        payload: Dict[str, Any] = {
            "From":      mobile,
            "CallerId":  EXOTEL_EXOPHONE,
            "Url":       flow_url,
            "CallType":  CALL_TYPE,
            "Record":    "true" if CALL_RECORD else "false",
            "TimeLimit": str(CALL_TIME_LIMIT_SECONDS),
            "TimeOut":   str(CALL_RING_TIMEOUT_SECONDS),

            # CustomField: pass the case_id so all Exotel webhooks can correlate
            # back to the correct NDR case in our database (max 128 chars)
            "CustomField": str(case_id),
        }

        # Attach status webhook if a domain is provided
        if webhook_domain:
            webhook_domain = webhook_domain.rstrip("/")
            payload["StatusCallback"] = f"{webhook_domain}/webhooks/exotel/status"
            payload["StatusCallbackContentType"] = STATUS_CALLBACK_CONTENT_TYPE

        logger.info(
            f"Initiating Exotel call | case_id={case_id} | mobile={mobile} "
            f"| caller_id={EXOTEL_EXOPHONE} | flow={flow_url}"
        )
        if webhook_domain:
            logger.info(f"StatusCallback → {payload['StatusCallback']}")

        # Execute API call with Basic Auth
        auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN)
        with httpx.Client(timeout=30.0) as client:
            response = client.post(api_url, auth=auth, data=payload)

        if response.status_code not in (200, 201):
            logger.error(
                f"Exotel call initiation failed | status={response.status_code} "
                f"| body={response.text}"
            )
            raise Exception(
                f"Exotel API error {response.status_code}: {response.text}"
            )

        result = response.json()
        call_sid = result.get("Call", {}).get("Sid", "unknown")
        call_status = result.get("Call", {}).get("Status", "unknown")

        logger.info(
            f"Exotel call queued successfully | call_sid={call_sid} "
            f"| status={call_status} | case_id={case_id}"
        )
        return result

    @staticmethod
    def get_call_details(call_sid: str) -> Dict[str, Any]:
        """
        Fetches the live or final details of a specific Exotel call by its SID.

        Args:
            call_sid (str): The Exotel Call SID returned during initiation

        Returns:
            dict: Parsed JSON response with call details (status, duration, recording URL, etc.)

        Raises:
            ValueError: If credentials are missing
            Exception: If the API returns a non-success status code
        """
        _validate_credentials()

        url = f"https://{EXOTEL_API_HOST}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls/{call_sid}.json"

        logger.info(f"Fetching Exotel call details | call_sid={call_sid}")

        auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN)
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url, auth=auth)

        if response.status_code != 200:
            logger.error(
                f"Exotel get_call_details failed | status={response.status_code} "
                f"| body={response.text}"
            )
            raise Exception(
                f"Exotel get_call_details API error {response.status_code}: {response.text}"
            )

        result = response.json()
        logger.info(
            f"Exotel call details fetched | call_sid={call_sid} "
            f"| status={result.get('Call', {}).get('Status')}"
        )
        return result
