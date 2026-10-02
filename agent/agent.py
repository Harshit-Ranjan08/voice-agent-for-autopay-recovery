import os
import re
import time
import logging
from typing import Optional, Literal
from datetime import datetime, timedelta
import dateparser

from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.state import AgentState
from database.connection import SessionLocal
from database.models import AutopayCase, CallAttempt
import database.crud as crud

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent.autopay")

# ==========================================
# HYPERPARAMETERS
# ==========================================
MAX_SILENCE_RETRIES = 3  # nudge1, nudge2, then close at retry3
SILENCE_TIMEOUT = 6      # seconds
MAX_CLARIFICATION_RETRIES = 2


# ==========================================
# STRUCTURED OUTPUT SCHEMAS (State-Delta Pattern)
# ==========================================
class SlotUpdates(BaseModel):
    agreed_retry_date: Optional[str] = Field(
        default=None,
        description="Retry or re-attempt date mentioned by customer (e.g. 'kal' -> 'tomorrow', 'Friday', '2026-10-05')."
    )
    callback_time: Optional[str] = Field(
        default=None,
        description="Callback time specified by customer (e.g. '5 PM', 'tomorrow morning 11am', 'shaam ko')."
    )
    dispute_reason: Optional[str] = Field(
        default=None,
        description="Reason customer is disputing the charge or claiming payment already done."
    )


class StateDelta(BaseModel):
    intent: Literal[
        "none",
        "identity_confirmed",
        "wrong_number",
        "language_barrier",
        "quick_pitch",
        "continue",
        "pay_now",
        "retry_autopay",
        "update_method",
        "callback",
        "dispute",
        "cancel_service",
        "query_details",
        "off_topic",
        "confirm",
        "deny",
        "unknown"
    ] = Field(
        default="none",
        description="Customer's detected intent in their latest utterance for the autopay recovery flow."
    )
    query_type: Optional[Literal["amount", "service", "reason", "method", "link", "all"]] = Field(
        default=None,
        description="Specific question asked: 'amount' (due amount), 'service' (plan or service name), 'reason' (why autopay failed), 'method' (which card/bank), 'link' (how to pay), or 'all'."
    )
    intent_confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0 of the intent extraction."
    )
    intent_changed: bool = Field(
        default=False,
        description="True if customer explicitly changed their mind/intent from a previous choice."
    )
    confirmation: Optional[Literal["yes", "no", "unknown"]] = Field(
        default=None,
        description="Yes/No response when agent asked a confirmation question."
    )
    detected_language: Optional[str] = Field(
        default=None,
        description="Language the customer used or requested: 'English' or 'Hindi' / 'Hinglish'."
    )
    identity_status: Optional[bool] = Field(
        default=None,
        description="True if customer confirmed their identity ('Yes', 'Speaking', 'Haan'). False if wrong number ('wrong person', 'nahi galat number')."
    )
    slot_updates: Optional[SlotUpdates] = Field(
        default=None,
        description="Extracted date/time or dispute slots from utterance."
    )
    customer_sentiment: Optional[Literal["Cooperative", "Frustrated", "Neutral", "Uninterested"]] = Field(
        default="Neutral",
        description="Detected sentiment of the customer."
    )
    is_off_topic: bool = Field(
        default=False,
        description="True if customer asked an unrelated question."
    )

    @field_validator("identity_status", mode="before")
    @classmethod
    def parse_identity_status(cls, v):
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes", "confirmed"):
                return True
            if v.lower() in ("false", "0", "no", "denied"):
                return False
            return None
        return v

    @field_validator("customer_sentiment", mode="before")
    @classmethod
    def parse_sentiment(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().capitalize()
            if v_clean in ["Cooperative", "Frustrated", "Neutral", "Uninterested"]:
                return v_clean
            return "Neutral"
        return v or "Neutral"

    @field_validator("intent", mode="before")
    @classmethod
    def parse_intent(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


# ==========================================
# LLM SETUP (Groq LPU / Gemini Fallback)
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    logger.info("[LLM SETUP] Using Groq LPU (openai/gpt-oss-120b) for low-latency NLU reasoning.")
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=500
    )
    structured_llm = llm.with_structured_output(StateDelta, method="json_mode")
else:
    logger.info("[LLM SETUP] Using Google Gemini 2.5 Flash for NLU reasoning.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=500,
    )
    structured_llm = llm.with_structured_output(StateDelta, method="json_mode")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _lang(hindi_text: str, english_text: str, language: str) -> str:
    return hindi_text if (language or "English").lower() in ["hindi", "hinglish"] else english_text


def format_amount(amount: float, currency: str = "INR") -> str:
    curr_symbol = "₹" if currency == "INR" else currency
    return f"{curr_symbol}{amount:,.2f}"


def format_failure_reason(reason: Optional[str], language: str) -> str:
    if not reason:
        return _lang("बैंक से पेमेंट प्रोसेस नहीं हो पाया", "a bank processing decline", language)
    r = reason.lower()
    if "insufficient" in r or "balance" in r or "funds" in r:
        return _lang("अकाउंट में इनसफिशिएंट बैलेंस की वजह से", "insufficient balance in your account", language)
    elif "expired" in r or "card" in r:
        return _lang("कार्ड एक्सपायर होने की वजह से", "the card being expired", language)
    elif "limit" in r:
        return _lang("ऑटोपे मैंडेट लिमिट एक्सीड होने के कारण", "mandate transaction limit exceeded", language)
    elif "dormant" in r or "inactive" in r:
        return _lang("अकाउंट इनएक्टिव होने की वजह से", "the bank account being inactive", language)
    elif "stop" in r:
        return _lang("स्टॉप पेमेंट रिक्वेस्ट के कारण", "a stop-payment instruction", language)
    else:
        return _lang("बैंक सर्वर टाइमआउट के कारण", "a technical bank server timeout", language)


# ==========================================
# NODE 1: load_context_node
# ==========================================
def load_context_node(state: AgentState) -> dict:
    """
    Loads Autopay case data from database (if available) and initializes state.
    """
    case_id = state.get("case_id")
    # If case context is already populated in state (e.g. simulator/tests), keep it
    if state.get("customer_name") and state.get("customer_name") != "Customer":
        return {
            "current_step": state.get("current_step") or "greeting",
            "current_language": state.get("current_language") or "English",
            "detected_intent": state.get("detected_intent", "none"),
            "silence_count": state.get("silence_count", 0)
        }

    if not case_id:
        return {}

    logger.info(f"[load_context] Loading context for case_id={case_id} step={state.get('current_step', 'greeting')}")

    try:
        db = SessionLocal()
        try:
            case = crud.get_autopay_case(db=db, case_id=case_id)
            if case:
                is_cb = (case.final_disposition == "Callback Requested")
                return {
                    "customer_id": state.get("customer_id") or case.customer_id or "",
                    "customer_name": state.get("customer_name") or case.customer_name or "Customer",
                    "mobile": state.get("mobile") or case.mobile or "",
                    "email": state.get("email") or case.email or "",
                    "service_type": state.get("service_type") or case.service_type or "Subscription",
                    "plan_or_loan_name": state.get("plan_or_loan_name") or case.plan_or_loan_name or "your service",
                    "due_amount": float(state.get("due_amount", case.due_amount or 0.0)),
                    "currency": state.get("currency") or case.currency or "INR",
                    "due_date": state.get("due_date") or (case.due_date.strftime("%Y-%m-%d") if case.due_date else ""),
                    "failed_date": state.get("failed_date") or (case.failed_date.strftime("%Y-%m-%d") if case.failed_date else ""),
                    "payment_method": state.get("payment_method") or case.payment_method or "Autopay Mandate",
                    "failure_reason": state.get("failure_reason") or case.failure_reason or "Bank Decline",
                    "payment_link": state.get("payment_link") or case.payment_link or f"https://rzp.io/i/rec_{case.id}",
                    "is_callback_retry": state.get("is_callback_retry", is_cb),
                    "current_step": state.get("current_step") or "greeting",
                    "current_language": state.get("current_language") or "English",
                    "is_identity_confirmed": state.get("is_identity_confirmed"),
                    "detected_intent": state.get("detected_intent", "none"),
                    "previous_intent": state.get("previous_intent"),
                    "intent_changed": state.get("intent_changed", False),
                    "intent_confidence": state.get("intent_confidence", 0.0),
                    "agreed_retry_date": state.get("agreed_retry_date"),
                    "callback_time": state.get("callback_time"),
                    "dispute_reason": state.get("dispute_reason"),
                    "final_disposition": state.get("final_disposition"),
                    "customer_sentiment": state.get("customer_sentiment", "Neutral"),
                    "payment_link_sent": state.get("payment_link_sent", False),
                    "silence_count": state.get("silence_count", 0)
                }
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[load_context] DB connection not available ({e}), using in-memory state.")

    return {
        "current_step": state.get("current_step") or "greeting",
        "current_language": state.get("current_language") or "English",
        "detected_intent": state.get("detected_intent", "none"),
        "silence_count": state.get("silence_count", 0)
    }


# ==========================================
# NODE 2: greeting_node
# ==========================================
def greeting_node(state: AgentState) -> dict:
    """
    Sends initial identity greeting.
    """
    customer_name = state.get("customer_name", "Customer")
    service_type = state.get("service_type", "Account")
    language = state.get("current_language", "English")

    logger.info(f"[greeting] Sending initial identity greeting | lang={language}")
    greeting_text = _lang(
        f"नमस्ते {customer_name} जी! मैं Razorpay Autopay टीम से बात कर रही हूँ। क्या मेरी बात {customer_name} जी से हो रही है?",
        f"Hello, good day! This is Arya calling from Razorpay regarding your {service_type} payment. Am I speaking with {customer_name}?",
        language
    )
    return {
        "messages": [AIMessage(content=greeting_text)],
        "current_step": "greeting",
        "silence_count": state.get("silence_count", 0)
    }


# ==========================================
# NODE 3: reasoning_node (NLU Engine & Silence Tracker)
# ==========================================
def reasoning_node(state: AgentState) -> dict:
    """
    State-Delta NLU Engine. Detects customer intents, extracts slots, handles silence nudges.
    """
    messages = state.get("messages", [])
    current_step = state.get("current_step", "greeting")
    language = state.get("current_language", "English")
    customer_name = state.get("customer_name", "Customer")
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    service_type = state.get("service_type", "Subscription")
    plan_name = state.get("plan_or_loan_name", "Plan")
    failure_reason = state.get("failure_reason", "")

    latest_utterance = ""
    if messages and isinstance(messages[-1], HumanMessage):
        latest_utterance = messages[-1].content.strip()

    # Per-step LLM prompt guidance
    if current_step == "greeting":
        allowed_intents = "identity_confirmed, wrong_number, language_barrier, quick_pitch, callback, continue, unknown"
        examples_block = (
            "EXAMPLES FOR GREETING:\n"
            "- Utterance: 'Yes, speaking' -> {\"intent\": \"identity_confirmed\", \"identity_status\": true, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Haan bolo main hi hoon' -> {\"intent\": \"identity_confirmed\", \"identity_status\": true, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Wrong number, no one by that name here' -> {\"intent\": \"wrong_number\", \"identity_status\": false, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Who is calling and what is this about?' -> {\"intent\": \"quick_pitch\", \"intent_confidence\": 0.90}\n"
            "- Utterance: 'Please speak in Hindi' -> {\"intent\": \"language_barrier\", \"detected_language\": \"Hindi\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'I am driving right now, call me after 4 PM' -> {\"intent\": \"callback\", \"slot_updates\": {\"callback_time\": \"4 PM\"}, \"intent_confidence\": 0.95}"
        )
    elif current_step in ["reason_discovery", "resolution_offer"]:
        allowed_intents = "pay_now, retry_autopay, update_method, callback, dispute, cancel_service, query_details, off_topic, confirm, deny, unknown"
        examples_block = (
            "EXAMPLES FOR AUTOPAY RESOLUTION:\n"
            "- Utterance: 'Send me the payment link, I will pay right now' -> {\"intent\": \"pay_now\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Haan link bhej do SMS par' -> {\"intent\": \"pay_now\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'I did not have balance. Retry tomorrow' -> {\"intent\": \"retry_autopay\", \"slot_updates\": {\"agreed_retry_date\": \"tomorrow\"}, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Kal dobara try kar lo balance add kar dunga' -> {\"intent\": \"retry_autopay\", \"slot_updates\": {\"agreed_retry_date\": \"tomorrow\"}, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'My debit card is expired, how do I change it?' -> {\"intent\": \"update_method\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Card expire ho gaya naya card link karna hai' -> {\"intent\": \"update_method\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'I already paid this yesterday, check your records' -> {\"intent\": \"dispute\", \"slot_updates\": {\"dispute_reason\": \"already paid yesterday\"}, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'I want to cancel this subscription, do not charge me' -> {\"intent\": \"cancel_service\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'How much is the due amount?' -> {\"intent\": \"query_details\", \"query_type\": \"amount\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Why did my autopay fail?' -> {\"intent\": \"query_details\", \"query_type\": \"reason\", \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Call me back at 5 PM' -> {\"intent\": \"callback\", \"slot_updates\": {\"callback_time\": \"5 PM\"}, \"intent_confidence\": 0.95}"
        )
    elif current_step == "collect_details":
        allowed_intents = "retry_autopay, callback, dispute, pay_now, confirm, deny, unknown"
        examples_block = (
            "EXAMPLES FOR COLLECT DETAILS:\n"
            "- Utterance: 'Tomorrow 2 PM' -> {\"intent\": \"callback\", \"slot_updates\": {\"callback_time\": \"tomorrow 2 PM\"}, \"intent_confidence\": 0.95}\n"
            "- Utterance: 'Day after tomorrow on Friday' -> {\"intent\": \"retry_autopay\", \"slot_updates\": {\"agreed_retry_date\": \"Friday\"}, \"intent_confidence\": 0.95}"
        )
    else:
        allowed_intents = "pay_now, retry_autopay, update_method, callback, dispute, cancel_service, confirm, deny, unknown"
        examples_block = ""

    # Silence handling
    if not latest_utterance:
        silence_count = state.get("silence_count", 0) + 1
        if silence_count >= MAX_SILENCE_RETRIES:
            logger.info(f"[reasoning] Max silence retries reached ({silence_count}) -> closing")
            return {
                "silence_count": silence_count,
                "detected_intent": "no_answer",
                "current_step": "closing",
                "is_silence_turn": True
            }
        
        # Nudge 1 or Nudge 2
        nudge_text = _lang(
            f"हेलो {customer_name} जी, क्या आप मुझे सुन पा रहे हैं?",
            f"Hello {customer_name}, are you able to hear me?",
            language
        ) if silence_count == 1 else _lang(
            f"हेलो, क्या आप लाइन पर हैं? क्या हम ऑटोपे रिकवरी के बारे में बात कर सकते हैं?",
            f"Hello, are you still there? Should I send you the payment link to clear your {plan_name} autopay?",
            language
        )
        return {
            "silence_count": silence_count,
            "messages": [AIMessage(content=nudge_text)],
            "is_silence_turn": True
        }

    # Reset silence count when customer spoke
    system_prompt = f"""You are the NLU reasoning engine for Razorpay's Failed Autopay Recovery Voice Agent.
Context:
- Customer Name: {customer_name}
- Service / Bill: {service_type} ({plan_name})
- Due Amount: {due_amount}
- Current Step: {current_step}
- Failure Reason: {failure_reason}

Allowed Intents for this step: {allowed_intents}

{examples_block}

Extract the intent, identity_status, language preference, customer sentiment, and slot updates (agreed_retry_date, callback_time, dispute_reason) in JSON matching the schema."""

    user_prompt = f"Customer Utterance: \"{latest_utterance}\""

    try:
        delta: StateDelta = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        logger.info(f"[reasoning] Extracted Delta: intent={delta.intent} conf={delta.intent_confidence} slots={delta.slot_updates}")
    except Exception as e:
        logger.error(f"[reasoning] LLM Structured Extraction Failed: {e}. Falling back to default.")
        delta = StateDelta(intent="unknown", intent_confidence=0.5)

    # Route updates
    new_step = current_step
    detected_intent = delta.intent

    if delta.detected_language:
        state["current_language"] = delta.detected_language

    if current_step == "greeting":
        if delta.identity_status is True or detected_intent in ["identity_confirmed", "continue", "confirm"]:
            new_step = "reason_discovery"
        elif delta.identity_status is False or detected_intent == "wrong_number":
            new_step = "wrong_number"
        elif detected_intent == "language_barrier":
            new_step = "language_handling"
        elif detected_intent == "quick_pitch":
            new_step = "quick_pitch"
        elif detected_intent == "callback":
            new_step = "callback_node"
        elif detected_intent == "cancel_service":
            new_step = "cancel_service"
        else:
            new_step = "reason_discovery"
    else:
        # Subsequent steps
        if detected_intent == "pay_now" or (detected_intent == "confirm" and current_step == "resolution_offer"):
            new_step = "pay_now"
        elif detected_intent == "retry_autopay":
            new_step = "retry_autopay"
        elif detected_intent == "update_method":
            new_step = "update_method"
        elif detected_intent == "callback":
            new_step = "callback_node"
        elif detected_intent == "dispute":
            new_step = "dispute_node"
        elif detected_intent == "cancel_service":
            new_step = "cancel_service"
        elif detected_intent == "query_details":
            new_step = "query_details"
        elif detected_intent == "language_barrier":
            new_step = "language_handling"
        elif detected_intent == "off_topic":
            new_step = "off_topic"
        elif detected_intent == "wrong_number":
            new_step = "wrong_number"
        elif detected_intent == "unknown":
            new_step = "clarification"

    # Extract slots
    agreed_date = delta.slot_updates.agreed_retry_date if delta.slot_updates else None
    cb_time = delta.slot_updates.callback_time if delta.slot_updates else None
    disp_reason = delta.slot_updates.dispute_reason if delta.slot_updates else None

    return {
        "detected_intent": detected_intent,
        "intent_confidence": delta.intent_confidence,
        "is_identity_confirmed": delta.identity_status if delta.identity_status is not None else state.get("is_identity_confirmed"),
        "current_step": new_step,
        "current_language": delta.detected_language or state.get("current_language", "English"),
        "agreed_retry_date": agreed_date or state.get("agreed_retry_date"),
        "callback_time": cb_time or state.get("callback_time"),
        "dispute_reason": disp_reason or state.get("dispute_reason"),
        "customer_sentiment": delta.customer_sentiment or state.get("customer_sentiment", "Neutral"),
        "silence_count": 0,
        "is_silence_turn": False
    }


# ==========================================
# STEP NODES
# ==========================================

def reason_discovery_node(state: AgentState) -> dict:
    """
    Informs the customer about the failed payment and offers resolution choices.
    """
    customer_name = state.get("customer_name", "Customer")
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    plan_name = state.get("plan_or_loan_name", "your plan")
    service_type = state.get("service_type", "subscription")
    failure_reason = format_failure_reason(state.get("failure_reason"), state.get("current_language", "English"))
    language = state.get("current_language", "English")

    text = _lang(
        f"धन्यवाद {customer_name} जी। मैं आपको सूचित करने के लिए कॉल कर रही हूँ कि आपके {plan_name} के लिए {due_amount} का ऑटोपे पेमेंट {failure_reason} पूरा नहीं हो सका। क्या मैं आपको अभी पेमेंट करने के लिए एसएमएस पर एक सुरक्षित पेमेंट लिंक भेज दूँ, या आप ऑटोपे दोबारा ट्राई करवाना चाहेंगे?",
        f"Thank you, {customer_name}. I'm calling to let you know that your scheduled autopay payment of {due_amount} for your {plan_name} was unsuccessful due to {failure_reason}. Would you like me to send you an instant secure payment link right now, or schedule an autopay retry?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "resolution_offer"
    }


def pay_now_node(state: AgentState) -> dict:
    """
    Handles immediate payment resolution by offering/confirming the payment link.
    """
    customer_name = state.get("customer_name", "Customer")
    mobile = state.get("mobile", "")
    last_4 = mobile[-4:] if len(mobile) >= 4 else mobile
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    language = state.get("current_language", "English")
    payment_link = state.get("payment_link", "https://rzp.io/pay")

    text = _lang(
        f"बहुत बढ़िया {customer_name} जी! मैंने आपके रजिस्टर्ड मोबाइल नंबर (आखिरी 4 अंक {last_4}) पर {due_amount} के लिए पेमेंट लिंक एसएमएस और व्हाट्सएप पर भेज दिया है। आप यूपीआई, डेबिट कार्ड या नेटबैंकिंग से तुरंत पेमेंट कर सकते हैं। क्या मैं आपकी कोई और मदद कर सकती हूँ?",
        f"Perfect, {customer_name}! I have triggered the secure payment link for {due_amount} to your registered mobile ending in {last_4} via SMS and WhatsApp. You can pay conveniently using UPI, Cards, or Netbanking. Is there anything else I can help you with today?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Payment Link Sent",
        "payment_link_sent": True
    }


def retry_autopay_node(state: AgentState) -> dict:
    """
    Handles scheduling an autopay retry on an agreed date.
    """
    customer_name = state.get("customer_name", "Customer")
    agreed_date = state.get("agreed_retry_date") or "tomorrow"
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    language = state.get("current_language", "English")

    text = _lang(
        f"ज़रूर {customer_name} जी! मैंने {due_amount} के लिए आपका ऑटोपे री-अटेम्प्ट {agreed_date} के लिए शेड्यूल कर दिया है। कृपया अपने अकाउंट में पर्याप्त बैलेंस बनाए रखें ताकि पेमेंट सफलतापूर्वक हो जाए। धन्यवाद और आपका दिन शुभ हो!",
        f"Understood, {customer_name}! I have scheduled the autopay re-attempt for {due_amount} on {agreed_date}. Please ensure sufficient balance is maintained in your linked account. Thank you and have a great day!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Autopay Retry Scheduled"
    }


def update_method_node(state: AgentState) -> dict:
    """
    Handles mandate update / expired card flow.
    """
    customer_name = state.get("customer_name", "Customer")
    mobile = state.get("mobile", "")
    last_4 = mobile[-4:] if len(mobile) >= 4 else mobile
    language = state.get("current_language", "English")

    text = _lang(
        f"कोई बात नहीं {customer_name} जी! मैंने आपके रजिस्टर्ड नंबर {last_4} पर नया कार्ड या यूपीआई मैंडेट लिंक करने के लिए एक सुरक्षित लिंक भेज दिया है। कृपया उस लिंक पर जाकर अपना नया पेमेंट मेथड अपडेट कर लें। धन्यवाद!",
        f"No problem at all, {customer_name}! I have sent a secure link to your mobile ending in {last_4} to update your autopay mandate with your new card or bank account. Please update it at your earliest convenience. Thank you!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Payment Method Update Requested"
    }


def callback_node(state: AgentState) -> dict:
    """
    Handles callback scheduling when customer is busy.
    """
    customer_name = state.get("customer_name", "Customer")
    cb_time = state.get("callback_time") or "a later time today"
    language = state.get("current_language", "English")

    text = _lang(
        f"बिल्कुल {customer_name} जी! मैंने आपके लिए {cb_time} पर कॉलबैक शेड्यूल कर दिया है। हमारी टीम आपसे उस समय संपर्क करेगी। धन्यवाद!",
        f"Certainly, {customer_name}! I have noted and scheduled a callback for you at {cb_time}. Our team will connect with you then. Have a wonderful day!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Callback Requested"
    }


def dispute_node(state: AgentState) -> dict:
    """
    Handles payment dispute or claims of already paid.
    """
    customer_name = state.get("customer_name", "Customer")
    language = state.get("current_language", "English")

    text = _lang(
        f"मैं आपकी बात समझ गई {customer_name} जी। मैंने आपकी रिक्वेस्ट और पेमेंट स्टेटस का नोट हमारे सपोर्ट सिस्टम में दर्ज कर दिया है। हमारी टीम 24 घंटे के अंदर बैंक से वेरीफाई करके आपको अपडेट कर देगी।",
        f"I understand, {customer_name}. I have recorded your dispute note in our support system. Our team will verify this with the bank and update your account within 24 hours. Thank you for informing us.",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Dispute Raised"
    }


def cancel_service_node(state: AgentState) -> dict:
    """
    Handles cancellation request.
    """
    customer_name = state.get("customer_name", "Customer")
    plan_name = state.get("plan_or_loan_name", "service")
    language = state.get("current_language", "English")

    text = _lang(
        f"मैंने {plan_name} के कैंसिलेशन की आपकी रिक्वेस्ट दर्ज कर ली है {customer_name} जी। हमारी कस्टमर सपोर्ट टीम आपसे फॉर्मेलिटी पूरी करने के लिए संपर्क करेगी। धन्यवाद!",
        f"I have recorded your request to cancel your {plan_name}, {customer_name}. Our customer support team will follow up with you to process the cancellation. Thank you!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Cancellation Requested"
    }


def query_details_node(state: AgentState) -> dict:
    """
    Answers questions regarding amount, bill details, failure reasons.
    """
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    plan_name = state.get("plan_or_loan_name", "Plan")
    payment_method = state.get("payment_method", "Autopay Mandate")
    failure_reason = format_failure_reason(state.get("failure_reason"), state.get("current_language", "English"))
    language = state.get("current_language", "English")

    text = _lang(
        f"आपके {plan_name} का कुल ड्यू अमाउंट {due_amount} है, जो {payment_method} से {failure_reason} प्रोसेस नहीं हो पाया था। क्या आप इसे अभी क्लियर करने के लिए पेमेंट लिंक चाहते हैं?",
        f"The pending amount for your {plan_name} is {due_amount}, which was declined via {payment_method} due to {failure_reason}. Would you like me to send you the instant payment link to clear this?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "resolution_offer"
    }


def wrong_number_node(state: AgentState) -> dict:
    language = state.get("current_language", "English")
    text = _lang(
        "असुविधा के लिए माफ़ी चाहती हूँ। मैं इस नंबर को हमारे रिकॉर्ड से अपडेट कर दूँगी। आपका दिन शुभ हो!",
        "Apologies for the inconvenience. I will update our records immediately. Have a pleasant day!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "closing",
        "final_disposition": "Wrong Number"
    }


def language_handling_node(state: AgentState) -> dict:
    language = state.get("current_language", "English")
    text = _lang(
        "हाँ बिल्कुल! हम हिंदी में बात कर सकते हैं।",
        "Sure, we can certainly continue in English." if language == "English" else "हाँ बिल्कुल, हम हिंदी में बात कर सकते हैं।",
        language
    )
    # Append pitch
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    plan_name = state.get("plan_or_loan_name", "plan")
    pitch = _lang(
        f" आपके {plan_name} का {due_amount} का ऑटोपे पेमेंट पेंडिंग है। क्या मैं आपको पेमेंट लिंक भेज दूँ?",
        f" Your autopay payment of {due_amount} for {plan_name} is pending. Would you like me to send you the payment link?",
        language
    )
    return {
        "messages": [AIMessage(content=text + pitch)],
        "current_step": "resolution_offer"
    }


def quick_pitch_node(state: AgentState) -> dict:
    service_type = state.get("service_type", "Account")
    plan_name = state.get("plan_or_loan_name", "Plan")
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    language = state.get("current_language", "English")

    text = _lang(
        f"मैं Razorpay से कॉल कर रही हूँ आपके {plan_name} के {due_amount} के पेंडिंग ऑटोपे पेमेंट के संदर्भ में। क्या आप पेमेंट लिंक प्राप्त करना चाहते हैं?",
        f"I am calling from Razorpay regarding your pending autopay payment of {due_amount} for {plan_name}. Would you like me to send you an instant payment link?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "resolution_offer"
    }


def off_topic_node(state: AgentState) -> dict:
    due_amount = format_amount(state.get("due_amount", 0.0), state.get("currency", "INR"))
    plan_name = state.get("plan_or_loan_name", "Plan")
    language = state.get("current_language", "English")

    text = _lang(
        f"मैं मुख्य रूप से आपके {plan_name} के {due_amount} के ऑटोपे पेमेंट के सिलसिले में कॉल कर रही हूँ। क्या आप इसे अभी क्लियर करना चाहेंगे?",
        f"I am primarily calling to help resolve the {due_amount} pending autopay payment for your {plan_name}. Would you like to clear it now or schedule a retry?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "resolution_offer"
    }


def clarification_node(state: AgentState) -> dict:
    language = state.get("current_language", "English")
    text = _lang(
        "माफ़ कीजिए, मुझे आपकी आवाज़ स्पष्ट नहीं सुनाई दी। क्या आप कृपया दोबारा बता सकते हैं?",
        "I'm sorry, I couldn't quite hear that clearly. Could you please repeat that?",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": state.get("current_step", "resolution_offer")
    }


def closing_node(state: AgentState) -> dict:
    language = state.get("current_language", "English")
    text = _lang(
        "बात करने के लिए बहुत-बहुत धन्यवाद। आपका दिन शुभ हो!",
        "Thank you for your time and cooperation. Have a wonderful day ahead!",
        language
    )
    return {
        "messages": [AIMessage(content=text)],
        "current_step": "completed"
    }


# ==========================================
# NODE: intent_execution_node (Database Outcomes)
# ==========================================
def intent_execution_node(state: AgentState) -> dict:
    """
    Persists final disposition, call outcomes, and summaries to DB.
    """
    case_id = state.get("case_id")
    attempt_id = state.get("attempt_id")
    final_disp = state.get("final_disposition") or "Completed"
    sentiment = state.get("customer_sentiment", "Neutral")
    
    # Parse retry date or callback datetime
    agreed_retry_dt = None
    cb_dt = None
    
    if state.get("agreed_retry_date"):
        try:
            agreed_retry_dt = dateparser.parse(state.get("agreed_retry_date"))
        except Exception:
            pass
            
    if state.get("callback_time"):
        try:
            cb_dt = dateparser.parse(state.get("callback_time"))
        except Exception:
            pass

    payment_status = "UNPAID"
    if final_disp == "Payment Link Sent":
        payment_status = "LINK_SENT"
    elif final_disp == "Autopay Retry Scheduled":
        payment_status = "RETRY_SCHEDULED"
    elif final_disp == "Dispute Raised":
        payment_status = "DISPUTED"
    elif final_disp == "Cancellation Requested":
        payment_status = "CANCELLED"

    notes = f"Disposition: {final_disp} | Sentiment: {sentiment}"
    if state.get("dispute_reason"):
        notes += f" | Dispute: {state.get('dispute_reason')}"

    logger.info(f"[intent_execution] Outcome determined | case_id={case_id} disp='{final_disp}' payment_status='{payment_status}'")

    try:
        db = SessionLocal()
        try:
            if attempt_id:
                crud.update_call_attempt_outcome(
                    db=db,
                    attempt_id=attempt_id,
                    status="COMPLETED",
                    final_disposition=final_disp,
                    notes=notes
                )
            
            if case_id:
                crud.update_autopay_case_status(
                    db=db,
                    case_id=case_id,
                    agent_status="COMPLETED",
                    final_disposition=final_disp,
                    call_summary=notes,
                    customer_sentiment=sentiment,
                    agreed_retry_date=agreed_retry_dt,
                    callback_scheduled_at=cb_dt,
                    payment_status=payment_status,
                    conclusion_date=datetime.utcnow()
                )
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"[intent_execution] DB write skipped/unavailable ({e})")

    return {"current_step": "completed"}


# ==========================================
# ROUTING LOGIC
# ==========================================
def route_by_step(state: AgentState) -> str:
    messages = state.get("messages", [])
    current_step = state.get("current_step", "greeting")

    if not messages:
        return "greeting"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage):
        return END

    if current_step in ["completed", "closing"]:
        return END

    silence_count = state.get("silence_count", 0)
    if silence_count >= MAX_SILENCE_RETRIES:
        return "closing"

    return "reasoning"


def route_after_reasoning(state: AgentState) -> str:
    if state.get("is_silence_turn"):
        return "closing" if state.get("current_step") == "closing" else "end_turn"

    step = state.get("current_step", "reason_discovery")
    valid_nodes = [
        "greeting",
        "reason_discovery",
        "pay_now",
        "retry_autopay",
        "update_method",
        "callback_node",
        "dispute_node",
        "cancel_service",
        "query_details",
        "wrong_number",
        "language_handling",
        "quick_pitch",
        "off_topic",
        "clarification",
        "closing",
        "completed"
    ]
    if step in valid_nodes:
        return step
    return "reason_discovery"


# ==========================================
# LANGGRAPH WORKFLOW BUILDER
# ==========================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("load_context", load_context_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("reasoning", reasoning_node)

workflow.add_node("reason_discovery", reason_discovery_node)
workflow.add_node("pay_now", pay_now_node)
workflow.add_node("retry_autopay", retry_autopay_node)
workflow.add_node("update_method", update_method_node)
workflow.add_node("callback_node", callback_node)
workflow.add_node("dispute_node", dispute_node)
workflow.add_node("cancel_service", cancel_service_node)
workflow.add_node("query_details", query_details_node)
workflow.add_node("wrong_number", wrong_number_node)
workflow.add_node("language_handling", language_handling_node)
workflow.add_node("quick_pitch", quick_pitch_node)
workflow.add_node("off_topic", off_topic_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("closing", closing_node)
workflow.add_node("intent_execution", intent_execution_node)

# Set Entry Point
workflow.set_entry_point("load_context")

workflow.add_conditional_edges(
    "load_context",
    route_by_step,
    {
        "greeting": "greeting",
        "reasoning": "reasoning",
        "closing": "closing",
        END: END
    }
)

workflow.add_conditional_edges(
    "reasoning",
    route_after_reasoning,
    {
        "greeting": "greeting",
        "reason_discovery": "reason_discovery",
        "pay_now": "pay_now",
        "retry_autopay": "retry_autopay",
        "update_method": "update_method",
        "callback_node": "callback_node",
        "dispute_node": "dispute_node",
        "cancel_service": "cancel_service",
        "query_details": "query_details",
        "wrong_number": "wrong_number",
        "language_handling": "language_handling",
        "quick_pitch": "quick_pitch",
        "off_topic": "off_topic",
        "clarification": "clarification",
        "closing": "closing",
        "completed": "intent_execution",
        "end_turn": END
    }
)

# Step Nodes -> END (waiting for customer turn)
workflow.add_edge("greeting", END)
workflow.add_edge("reason_discovery", END)
workflow.add_edge("language_handling", END)
workflow.add_edge("quick_pitch", END)
workflow.add_edge("query_details", END)
workflow.add_edge("off_topic", END)
workflow.add_edge("clarification", END)

# Terminal / Action Nodes -> intent_execution -> END
workflow.add_edge("pay_now", "intent_execution")
workflow.add_edge("retry_autopay", "intent_execution")
workflow.add_edge("update_method", "intent_execution")
workflow.add_edge("callback_node", "intent_execution")
workflow.add_edge("dispute_node", "intent_execution")
workflow.add_edge("cancel_service", "intent_execution")
workflow.add_edge("wrong_number", "intent_execution")
workflow.add_edge("closing", "intent_execution")
workflow.add_edge("intent_execution", END)

# Compile LangGraph Agent
autopay_agent = workflow.compile()
ndr_agent = autopay_agent  # Backward compatibility alias
logger.info("Autopay LangGraph Recovery Agent compiled successfully.")