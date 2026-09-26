import os
import json
import requests
from utils.settings import BUSINESS_EMAIL

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
N8N_WEBHOOK_AUTH_TOKEN = os.getenv("N8N_WEBHOOK_AUTH_TOKEN", "").strip()


def _normalise_n8n_result(result):
    """Handle n8n's direct JSON response and common wrapped response shapes."""
    for _ in range(3):
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (TypeError, ValueError):
                return None
        if not isinstance(result, dict):
            return None
        nested = result.get("body")
        if nested is None:
            nested = result.get("data")
        if isinstance(nested, (dict, str)):
            result = nested
            continue
        break

    if not isinstance(result, dict):
        return None
    # Normalize common Google Calendar/n8n ID spellings for the CRM.
    event_id = (result.get("event_id") or result.get("eventId") or
                result.get("calendar_event_id") or result.get("id"))
    if event_id:
        result["event_id"] = event_id
    status = str(result.get("status", "")).strip().lower()
    if status in {"success", "ok", "created", "updated", "deleted"}:
        result["status"] = "success"
    elif status:
        result["status"] = status
    return result

def trigger_n8n_booking_workflow(intent: str, client_name: str, phone: str, city: str = None,
                                  target_area: str = None, meeting_time: str = None, email: str = None,
                                  event_id: str = None, property_title: str = None, assigned_employee_email: str = None,
                                  assigned_employee_name: str = None):
    if not N8N_WEBHOOK_URL:
        return {"status": "skipped", "message": "Optional n8n webhook is not configured."}
    if not N8N_WEBHOOK_AUTH_TOKEN:
        return {"status": "error", "message": "n8n webhook is configured but N8N_WEBHOOK_AUTH_TOKEN is missing."}
    payload = {
        "intent": intent,
        "client_name": client_name,
        "phone": phone,
        # Notifications are internal: route them to the employee assigned to this visit.
        "email": assigned_employee_email or BUSINESS_EMAIL,
        "assigned_employee_email": assigned_employee_email or BUSINESS_EMAIL,
        "assigned_employee_name": assigned_employee_name or "Assigned Agent",
        "city": city or "Lahore",
        "target_area": target_area or "DHA Phase 5",
        "property_title": property_title or "",
        "meeting_time": meeting_time,
        "event_id": event_id,
    }
    
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={"X-Uzma-Webhook-Token": N8N_WEBHOOK_AUTH_TOKEN},
            timeout=30,
        )
        response.raise_for_status()
        if not response.content or not response.text.strip():
            return {
                "status": "error",
                "message": f"n8n returned an empty response (HTTP {response.status_code}); check that the selected workflow branch reaches a Respond to Webhook node.",
            }
        try:
            result = _normalise_n8n_result(response.json())
        except ValueError:
            return {
                "status": "error",
                "message": f"n8n returned a non-JSON response (HTTP {response.status_code}).",
            }
        if not isinstance(result, dict) or result.get("status") != "success":
            logging_message = (
                "n8n response did not contain a success status or calendar event ID; "
                f"HTTP {response.status_code}, response keys="
                f"{sorted(result.keys()) if isinstance(result, dict) else type(result).__name__}"
            )
            print(logging_message)
            return {"status": "error", "message": (result.get("message") if isinstance(result, dict) else None) or "n8n did not confirm the requested action."}
        if not result.get("event_id"):
            return {"status": "error", "message": "n8n returned success without a Google Calendar event ID."}
        if intent in {"reschedule", "cancel"} and event_id and str(result["event_id"]) != str(event_id):
            return {"status": "error", "message": "n8n returned a different Calendar event than the requested appointment."}
        return result
    except Exception as e:
        print(f"Error calling n8n pipeline: {e}")
        return {"status": "error", "message": str(e)}
