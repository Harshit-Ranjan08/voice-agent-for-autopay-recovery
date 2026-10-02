import pandas as pd
import numpy as np
from datetime import datetime
import re
import io
from typing import List, Dict, Any, Union

def clean_phone(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, float):
        val = int(val)
    val_str = str(val).strip()
    val_str = re.sub(r"\.0$", "", val_str)
    val_str = "".join(filter(str.isdigit, val_str))
    return val_str

def clean_string(val) -> Union[str, None]:
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip()
    return val_str if val_str else None

def clean_int(val, default: int = 0) -> int:
    if pd.isna(val) or val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def clean_float(val, default: float = 0.0) -> float:
    if pd.isna(val) or val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def clean_date(val) -> Union[datetime, None]:
    if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", "nat", ""]:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None

def parse_autopay_file(file_path_or_bytes: Union[str, bytes, io.BytesIO], filename: str = "") -> List[Dict[str, Any]]:
    """
    Parses an Excel (.xlsx, .xls) or CSV file containing failed autopay customer records.
    Returns a list of clean dictionaries ready to be inserted into the database.
    """
    if isinstance(file_path_or_bytes, bytes):
        file_buffer = io.BytesIO(file_path_or_bytes)
    else:
        file_buffer = file_path_or_bytes

    # Determine parser based on filename or format
    is_csv = False
    if isinstance(filename, str) and filename.lower().endswith(".csv"):
        is_csv = True
    elif isinstance(file_path_or_bytes, str) and file_path_or_bytes.lower().endswith(".csv"):
        is_csv = True

    try:
        if is_csv:
            df = pd.read_csv(file_buffer)
        else:
            try:
                df = pd.read_excel(file_buffer)
            except Exception:
                # Fallback to CSV if excel read fails
                if hasattr(file_buffer, 'seek'):
                    file_buffer.seek(0)
                df = pd.read_csv(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read file into DataFrame: {e}")

    # Standardize column names (strip whitespace and convert to lowercase for lookup)
    normalized_cols = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=normalized_cols)

    def get_val(row, *aliases):
        for alias in aliases:
            cleaned_alias = alias.lower().replace(" ", "_")
            if cleaned_alias in row:
                return row.get(cleaned_alias)
        return None

    cases = []
    for _, row in df.iterrows():
        customer_id = clean_string(get_val(row, "customer_id", "cust_id", "lead_id", "id"))
        customer_name = clean_string(get_val(row, "customer_name", "name", "client_name"))
        mobile = clean_phone(get_val(row, "mobile", "phone", "phone_number", "contact"))
        email = clean_string(get_val(row, "email", "email_id", "mail"))
        service_type = clean_string(get_val(row, "service_type", "service", "type", "category")) or "Subscription"
        plan_or_loan_name = clean_string(get_val(row, "plan_or_loan_name", "plan_name", "loan_name", "product", "item"))
        due_amount = clean_float(get_val(row, "due_amount", "amount", "failed_amount", "due_val", "collectable"), 0.0)
        currency = clean_string(get_val(row, "currency")) or "INR"
        due_date = clean_date(get_val(row, "due_date", "payment_due_date", "date"))
        failed_date = clean_date(get_val(row, "failed_date", "failure_date", "last_attempt_date"))
        payment_method = clean_string(get_val(row, "payment_method", "mandate_type", "mode", "payment_mode"))
        failure_reason = clean_string(get_val(row, "failure_reason", "reason", "error_reason", "last_failed_reason"))
        payment_link = clean_string(get_val(row, "payment_link", "pay_url", "link", "url"))
        payment_status = clean_string(get_val(row, "payment_status", "status")) or "UNPAID"

        # Generate fallback payment link if not provided
        if not payment_link and customer_id:
            payment_link = f"https://rzp.io/i/rec_{customer_id.lower().replace('-', '_')}"

        case_data = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "mobile": mobile,
            "email": email,
            "service_type": service_type,
            "plan_or_loan_name": plan_or_loan_name,
            "due_amount": due_amount,
            "currency": currency,
            "due_date": due_date,
            "failed_date": failed_date,
            "payment_method": payment_method,
            "failure_reason": failure_reason,
            "payment_link": payment_link,
            "payment_status": payment_status,
            "attempt_count": 0
        }

        # Keep record if name and mobile are present
        if case_data["customer_name"] and case_data["mobile"]:
            cases.append(case_data)

    return cases


# Backward compatibility alias
parse_excel_report = parse_autopay_file
