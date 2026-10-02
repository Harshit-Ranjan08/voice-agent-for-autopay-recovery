from sqlalchemy import Column, Integer, BigInteger, String, Text, Numeric, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import relationship
from database.connection import Base

class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=True)
    total_records = Column(Integer, nullable=False, default=0)
    status = Column(String(50), default="UPLOADED")  # UPLOADED, PROCESSING, READY, CALLING, COMPLETED, FAILED
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship to autopay recovery cases
    cases = relationship("AutopayCase", back_populates="campaign", cascade="all, delete-orphan")


class AutopayCase(Base):
    __tablename__ = "autopay_cases"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    campaign_id = Column(BigInteger, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    customer_id = Column(String(100), nullable=True)  # e.g., CUST-1001
    customer_name = Column(String(255), nullable=False)
    mobile = Column(String(20), nullable=False)
    email = Column(String(255), nullable=True)
    
    # Financial / Subscription / Bill Details
    service_type = Column(String(100), nullable=False)  # Subscription, Loan EMI, Credit Card, Broadband, Insurance, SaaS
    plan_or_loan_name = Column(String(255), nullable=True)  # e.g., "Premium Annual Plan", "Personal Loan EMI #5"
    due_amount = Column(Numeric(10, 2), nullable=False, default=0.0)  # Failed Autopay Amount
    currency = Column(String(10), default="INR")
    due_date = Column(DateTime, nullable=True)  # Original payment due date
    failed_date = Column(DateTime, nullable=True)  # Date when autopay failed
    payment_method = Column(String(100), nullable=True)  # e.g., HDFC Debit Card e-Mandate, ICICI eNACH, UPI Autopay
    failure_reason = Column(String(255), nullable=True)  # e.g., Insufficient Balance, Expired Card, Mandate Limit Exceeded
    payment_link = Column(String(500), nullable=True)  # Instant payment link generated for the customer
    
    # Attempt Tracking
    attempt_count = Column(Integer, default=0)
    last_attempt_date = Column(DateTime, nullable=True)
    
    # AI Voice Workflow & Disposition
    agent_status = Column(String(50), default="PENDING")  # PENDING, CALLING, COMPLETED, FAILED
    final_disposition = Column(String(100), nullable=True)  # Payment Link Sent, Retry Scheduled, Method Update, Paid, Callback, Cancel/Dispute, No Answer
    call_summary = Column(Text, nullable=True)
    customer_sentiment = Column(String(50), nullable=True)  # Cooperative, Frustrated, Neutral, Uninterested
    agreed_retry_date = Column(DateTime, nullable=True)  # If customer requests autopay re-attempt on a specific date
    callback_scheduled_at = Column(DateTime, nullable=True)  # If customer requests a callback
    payment_status = Column(String(50), default="UNPAID")  # UNPAID, PAID, LINK_SENT, RETRY_SCHEDULED, DISPUTED, CANCELLED
    transaction_reference = Column(String(100), nullable=True)
    conclusion_date = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    campaign = relationship("Campaign", back_populates="cases")
    call_attempts = relationship("CallAttempt", back_populates="autopay_case", cascade="all, delete-orphan")


class CallAttempt(Base):
    __tablename__ = "call_attempts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    case_id = Column(BigInteger, ForeignKey("autopay_cases.id", ondelete="CASCADE"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    provider_call_id = Column(String(100), nullable=True)  # Telephony Call SID / Provider Reference
    call_direction = Column(String(20), default="OUTBOUND")
    call_status = Column(String(50), nullable=True)  # CONNECTED, NO_ANSWER, BUSY, FAILED, IN_PROGRESS
    call_outcome = Column(String(100), nullable=True)  # Disposition outcome detected for this call
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    minutes_spoken = Column(Numeric(5, 2), default=0.0)
    disconnected_by = Column(String(50), nullable=True)  # Customer, Agent, Provider
    feedback = Column(Text, nullable=True)
    recording_url = Column(Text, nullable=True)
    conversation_history = Column(JSON, nullable=True)  # List of dialogue turns
    
    created_at = Column(DateTime, default=func.now())

    # Relationships
    autopay_case = relationship("AutopayCase", back_populates="call_attempts")
