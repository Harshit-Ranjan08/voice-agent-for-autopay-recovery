# 📞 Autopay Recovery Voice Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-f55036.svg)](https://groq.com/)
[![Sarvam AI](https://img.shields.io/badge/Sarvam%20AI-Indic%20Speech-7c3aed.svg)](https://www.sarvam.ai/)
[![Exotel](https://img.shields.io/badge/Exotel-Telephony%20WebSocket-0284c7.svg)](https://exotel.com/)

An enterprise-grade **AI Voice Agent system** engineered to autonomously recover failed recurring autopay payments (subscriptions, loan EMIs, utility bills, and insurance premiums). Built on **LangGraph**, **Groq LPU / Gemini**, **Sarvam AI (STT/TTS)**, **Exotel Telephony**, and **FastAPI**.

---

## 📌 Problem Statement & Use Case

In subscription businesses, SaaS platforms, and lending institutions, **15–25% of recurring autopay transactions fail** due to reasons such as:
* Temporary insufficient balance
* Expired or reissued debit/credit cards
* Mandate transaction limits
* Bank server timeouts and technical decline
* Customer disputes or missed billing dates

Traditional automated SMS reminders have low conversion rates, and manual call centers are expensive and slow to scale.

### 💡 The Solution
This AI Voice Agent:
1. Proactively calls customers when an autopay transaction fails.
2. Clearly explains the failed amount, billing service, and failure reason.
3. Understands customer intent and objections in real-time (English, Hindi, and Hinglish).
4. Executes dynamic resolution paths:
   * **Instant Payment**: Sends a secure SMS/WhatsApp payment link while on the call.
   * **Autopay Retry Scheduling**: Agrees on a future date to re-trigger the debit.
   * **Mandate Update**: Sends links to update expired cards or switch bank mandates.
   * **Smart Rescheduling**: Notes customer callback times when busy or driving.
   * **Dispute & Cancellation Handling**: Records disputes for 24-hour verification.

---

## 🏗️ Architecture & Conversation Flow

```mermaid
flowchart TD
    Start([📞 Outbound Call Triggered]) --> Greeting[1. Greeting & Identity Verification]
    Greeting -->|Confirmed| InformFailure[2. Inform Failed Autopay Amount & Reason]
    Greeting -->|Wrong Number| WrongNum[Exit: Wrong Number]
    Greeting -->|Busy / Driving| ScheduleCB[Schedule Callback Time]

    InformFailure --> Probe[3. Objection Discovery & Resolution]

    Probe -->|Wants to Pay Now| SendLink[Branch A: Send Instant Payment Link]
    Probe -->|Will maintain balance later| ScheduleRetry[Branch B: Schedule Autopay Re-attempt]
    Probe -->|Card Expired / Change Bank| MandateLink[Branch C: Send Mandate Update Link]
    Probe -->|Already Paid / Disputing| Dispute[Branch D: Log Dispute & Escalate]
    Probe -->|Call Later| ScheduleCB

    SendLink --> ConfirmLink[Confirm SMS/WhatsApp Sent]
    ConfirmLink --> WrapUp[4. Summary & Polite Farewell]
    ScheduleRetry --> ConfirmRetry[Confirm Re-attempt Date]
    ConfirmRetry --> WrapUp
    MandateLink --> WrapUp
    Dispute --> WrapUp
    ScheduleCB --> WrapUp

    WrapUp --> PersistDB[(💾 Persist Call Outcome & Dispositions to DB)]

