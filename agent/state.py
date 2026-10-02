from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # ==========================================
    # CONVERSATION HISTORY
    # ==========================================
    # Each turn appends to this list automatically using operator.add
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # ==========================================
    # AUTOPAY CASE CONTEXT (loaded from DB)
    # ==========================================
    case_id: int
    attempt_id: int                    # DB ID of the current call_attempt row
    customer_id: str                   # e.g., "CUST-1001"
    customer_name: str                 # e.g., "Aarav Sharma"
    mobile: str                        # e.g., "9876543210"
    email: Optional[str]               # e.g., "aarav.sharma@example.com"
    service_type: str                  # e.g., "SaaS Subscription", "Loan EMI", "Broadband", "Insurance"
    plan_or_loan_name: str             # e.g., "TechFlow Enterprise Cloud Monthly"
    due_amount: float                  # Failed Autopay Amount (e.g., 2499.0)
    currency: str                      # Default "INR"
    due_date: Optional[str]            # e.g., "2026-09-28"
    failed_date: Optional[str]         # e.g., "2026-09-29"
    payment_method: Optional[str]      # e.g., "HDFC Debit Card e-Mandate", "UPI Autopay"
    failure_reason: Optional[str]      # e.g., "Insufficient Balance", "Expired Card"
    payment_link: Optional[str]        # e.g., "https://rzp.io/i/rec_1001_aarav"
    is_callback_retry: bool            # True if this call is a callback retry

    # ==========================================
    # DIALOGUE FLOW STATE
    # ==========================================
    current_step: str
    # Values:
    # "greeting"           → introduces agent & confirms identity
    # "reason_discovery"   → informs about failed autopay & probes customer preference
    # "resolution_offer"   → presents solution (instant link, retry date, update mandate)
    # "collect_details"    → collects specific retry date / callback time / dispute note
    # "closing"            → summarizes agreement & polite farewell
    # "completed"          → call is fully concluded

    current_language: str              # "English" (default) or "Hindi" / "Hinglish"
    is_identity_confirmed: Optional[bool]  # True=confirmed, False=wrong number/person, None=pending

    # ==========================================
    # DETECTED INTENT & RESOLUTION PARAMETERS
    # ==========================================
    detected_intent: str
    # Values:
    # "none"                  → not yet determined
    # "pay_now"               → customer wants to pay immediately (requesting payment link)
    # "retry_autopay"         → customer wants autopay re-attempted on another date
    # "update_method"         → card expired / wants to switch bank/UPI mandate
    # "callback"              → customer is busy, requests callback at specific time
    # "dispute"               → customer disputes the charge or claims already paid
    # "cancel_service"        → customer wants to cancel subscription/service
    # "wrong_number"          → wrong person reached
    # "no_answer"             → silence / unreachable

    previous_intent: Optional[str]     # Prior intent before an intent change event
    intent_changed: bool               # True if customer explicitly changed mind/intent
    intent_confidence: float           # NLU confidence score (0.0 - 1.0)
    is_off_topic: bool                 # True if customer asked an off-topic question

    # Extracted Resolution Details:
    agreed_retry_date: Optional[str]   # e.g., "tomorrow", "2026-10-05", "Friday"
    callback_time: Optional[str]       # e.g., "5:00 PM", "tomorrow morning"
    dispute_reason: Optional[str]      # e.g., "already debited from bank yesterday"
    final_disposition: Optional[str]   # e.g., "Payment Link Sent", "Autopay Retry Scheduled", etc.
    customer_sentiment: Optional[str]  # "Cooperative", "Frustrated", "Neutral", "Uninterested"
    call_summary: Optional[str]        # Generated summary of the conversation
    payment_link_sent: bool            # True if payment link SMS was triggered during call

    # ==========================================
    # CALL & NUDGE METADATA
    # ==========================================
    silence_count: int                 # Number of consecutive empty/inaudible transcripts
    clarification_count: int           # Number of consecutive low-confidence turns
    is_silence_turn: bool              # True on turns where silence nudge was generated