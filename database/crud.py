from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional
from database.models import Campaign, AutopayCase, CallAttempt

# ==========================================
# CAMPAIGN CRUD
# ==========================================

def create_campaign(db: Session, name: str, file_name: str, file_path: str = None, total_records: int = 0) -> Campaign:
    db_campaign = Campaign(
        name=name,
        file_name=file_name,
        file_path=file_path,
        total_records=total_records,
        status="UPLOADED"
    )
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign


def get_campaign(db: Session, campaign_id: int) -> Optional[Campaign]:
    return db.query(Campaign).filter(Campaign.id == campaign_id).first()


def get_all_campaigns(db: Session) -> List[Campaign]:
    return db.query(Campaign).order_by(Campaign.created_at.desc()).all()


def update_campaign_status(db: Session, campaign_id: int, status: str) -> Optional[Campaign]:
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if campaign:
        campaign.status = status
        db.commit()
        db.refresh(campaign)
    return campaign


# ==========================================
# AUTOPAY CASE CRUD
# ==========================================

def create_autopay_case(db: Session, campaign_id: int, data: dict) -> AutopayCase:
    db_case = AutopayCase(
        campaign_id=campaign_id,
        customer_id=data.get("customer_id"),
        customer_name=data.get("customer_name"),
        mobile=str(data.get("mobile", "")),
        email=data.get("email"),
        service_type=data.get("service_type") or "Subscription",
        plan_or_loan_name=data.get("plan_or_loan_name"),
        due_amount=float(data.get("due_amount", 0.0)),
        currency=data.get("currency", "INR"),
        due_date=data.get("due_date"),
        failed_date=data.get("failed_date"),
        payment_method=data.get("payment_method"),
        failure_reason=data.get("failure_reason"),
        payment_link=data.get("payment_link"),
        attempt_count=data.get("attempt_count", 0),
        agent_status="PENDING",
        payment_status=data.get("payment_status", "UNPAID")
    )
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case


def get_autopay_case(db: Session, case_id: int) -> Optional[AutopayCase]:
    return db.query(AutopayCase).filter(AutopayCase.id == case_id).first()


def get_autopay_case_by_customer_id(db: Session, customer_id: str) -> Optional[AutopayCase]:
    return db.query(AutopayCase).filter(AutopayCase.customer_id == customer_id).first()


def get_autopay_case_by_mobile(db: Session, mobile: str) -> Optional[AutopayCase]:
    # Strip common prefixes for matching
    cleaned_mobile = mobile.replace("+91", "").replace(" ", "").strip()
    return db.query(AutopayCase).filter(AutopayCase.mobile.like(f"%{cleaned_mobile}%")).first()


def get_pending_autopay_cases(db: Session, campaign_id: Optional[int] = None, limit: int = 100) -> List[AutopayCase]:
    query = db.query(AutopayCase).filter(AutopayCase.agent_status == "PENDING")
    if campaign_id:
        query = query.filter(AutopayCase.campaign_id == campaign_id)
    return query.limit(limit).all()


def get_all_autopay_cases(db: Session, campaign_id: Optional[int] = None) -> List[AutopayCase]:
    query = db.query(AutopayCase)
    if campaign_id:
        query = query.filter(AutopayCase.campaign_id == campaign_id)
    return query.order_by(AutopayCase.id.asc()).all()


def update_autopay_case_status(
    db: Session,
    case_id: int,
    agent_status: str,
    final_disposition: Optional[str] = None,
    call_summary: Optional[str] = None,
    customer_sentiment: Optional[str] = None,
    agreed_retry_date: Optional[datetime] = None,
    callback_scheduled_at: Optional[datetime] = None,
    payment_status: Optional[str] = None,
    transaction_reference: Optional[str] = None,
    conclusion_date: Optional[datetime] = None
) -> Optional[AutopayCase]:
    case = db.query(AutopayCase).filter(AutopayCase.id == case_id).first()
    if case:
        case.agent_status = agent_status
        if final_disposition is not None:
            case.final_disposition = final_disposition
        if call_summary is not None:
            case.call_summary = call_summary
        if customer_sentiment is not None:
            case.customer_sentiment = customer_sentiment
        if agreed_retry_date is not None:
            case.agreed_retry_date = agreed_retry_date
        if callback_scheduled_at is not None:
            case.callback_scheduled_at = callback_scheduled_at
        if payment_status is not None:
            case.payment_status = payment_status
        if transaction_reference is not None:
            case.transaction_reference = transaction_reference
        if conclusion_date is not None:
            case.conclusion_date = conclusion_date
        
        db.commit()
        db.refresh(case)
    return case


def increment_case_attempt_count(db: Session, case_id: int) -> Optional[AutopayCase]:
    case = db.query(AutopayCase).filter(AutopayCase.id == case_id).first()
    if case:
        case.attempt_count = (case.attempt_count or 0) + 1
        case.last_attempt_date = datetime.utcnow()
        db.commit()
        db.refresh(case)
    return case


# ==========================================
# CALL ATTEMPT CRUD
# ==========================================

def create_call_attempt(
    db: Session,
    case_id: int,
    attempt_number: int,
    provider_call_id: Optional[str] = None
) -> CallAttempt:
    db_attempt = CallAttempt(
        case_id=case_id,
        attempt_number=attempt_number,
        provider_call_id=provider_call_id,
        call_direction="OUTBOUND",
        call_status="INITIATED"
    )
    db.add(db_attempt)
    db.commit()
    db.refresh(db_attempt)
    return db_attempt


def update_call_attempt(
    db: Session,
    attempt_id: int,
    call_status: Optional[str] = None,
    call_outcome: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    minutes_spoken: float = 0.0,
    disconnected_by: Optional[str] = None,
    feedback: Optional[str] = None,
    recording_url: Optional[str] = None,
    conversation_history: Optional[list] = None
) -> Optional[CallAttempt]:
    attempt = db.query(CallAttempt).filter(CallAttempt.id == attempt_id).first()
    if attempt:
        if call_status is not None:
            attempt.call_status = call_status
        if call_outcome is not None:
            attempt.call_outcome = call_outcome
        if start_time is not None:
            attempt.start_time = start_time
        if end_time is not None:
            attempt.end_time = end_time
        if minutes_spoken is not None:
            attempt.minutes_spoken = minutes_spoken
        if disconnected_by is not None:
            attempt.disconnected_by = disconnected_by
        if feedback is not None:
            attempt.feedback = feedback
        if recording_url is not None:
            attempt.recording_url = recording_url
        if conversation_history is not None:
            attempt.conversation_history = conversation_history
        db.commit()
        db.refresh(attempt)
    return attempt


def update_call_attempt_outcome(
    db: Session,
    attempt_id: int,
    status: str = "COMPLETED",
    final_disposition: Optional[str] = None,
    notes: Optional[str] = None
) -> Optional[CallAttempt]:
    attempt = db.query(CallAttempt).filter(CallAttempt.id == attempt_id).first()
    if attempt:
        attempt.call_status = status
        attempt.call_outcome = final_disposition
        attempt.feedback = notes
        attempt.end_time = datetime.utcnow()
        db.commit()
        db.refresh(attempt)
    return attempt


def get_call_attempts_for_case(db: Session, case_id: int) -> List[CallAttempt]:
    return db.query(CallAttempt).filter(CallAttempt.case_id == case_id).order_by(CallAttempt.attempt_number.asc()).all()


def get_callback_retry_cases(db: Session) -> List[AutopayCase]:
    """
    Fetches Autopay cases that are marked for Callback and scheduled for now/earlier,
    OR cases that failed/no_answer with total attempts < 3.
    """
    now = datetime.utcnow()
    # 1. Scheduled callbacks due
    callbacks = db.query(AutopayCase).filter(
        AutopayCase.final_disposition == "Callback",
        AutopayCase.callback_scheduled_at <= now
    ).all()
    
    # 2. Failed / No Answer cases with < 3 attempts
    failed_cases = db.query(AutopayCase).filter(
        AutopayCase.agent_status.in_(["FAILED", "NO_ANSWER"])
    ).all()
    
    retriable_failed = []
    for case in failed_cases:
        attempts_count = db.query(CallAttempt).filter(CallAttempt.case_id == case.id).count()
        if attempts_count < 3:
            retriable_failed.append(case)
            
    return list(set(callbacks + retriable_failed))
