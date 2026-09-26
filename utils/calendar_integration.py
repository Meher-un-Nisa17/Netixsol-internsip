import os
import datetime
import logging
import shutil
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

class GoogleCalendarManager:
    """Manages Google Calendar API interactions for scheduling, rescheduling, and cancellations."""
    
    def __init__(self, token_path=None):
        self.service = None
        self._authenticate(token_path or os.getenv("GOOGLE_TOKEN_PATH", "token.json"))

    def _authenticate(self, token_path):
        try:
            secret_path = os.getenv("GOOGLE_TOKEN_SECRET_PATH")
            if not os.path.exists(token_path) and secret_path and os.path.isfile(secret_path):
                os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
                shutil.copyfile(secret_path, token_path)
                try:
                    os.chmod(token_path, 0o600)
                except OSError:
                    pass
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/calendar'])
                
                # Expired token ko auto-refresh karna
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    with open(token_path, 'w') as token_file:
                        token_file.write(creds.to_json())

                if creds and creds.valid:
                    self.service = build('calendar', 'v3', credentials=creds)
                else:
                    logging.error("❌ Google Calendar Credentials valid nahi hain.")
            else:
                logging.error(f"❌ Token file missing at path: {token_path}")
        except Exception as e:
            logging.error(f"❌ Calendar Authentication Failure: {e}")

    def _calculate_start_and_end_iso(self, date_time_str: str, duration_minutes: int = 60):
        """Date/Time parse karke Start aur End times (with duration) banata hai."""
        clean_str = date_time_str.strip().replace(" ", "T")
        
        # Datetime object mein convert karna
        if len(clean_str) == 16:  # e.g., "2026-10-20T14:00"
            start_dt = datetime.datetime.strptime(clean_str, "%Y-%m-%dT%H:%M")
        elif len(clean_str) == 19:  # e.g., "2026-10-20T14:00:00"
            start_dt = datetime.datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")
        else:
            start_dt = datetime.datetime.fromisoformat(clean_str)

        # Meeting ki duration add karna (Default: 1 Hour)
        end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)
        
        # ISO format with timezone (+05:00 PKT)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00")
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S+05:00")
        
        return start_iso, end_iso

    def is_slot_available(self, date_time_str: str, duration_minutes: int = 60, exclude_event_id: str = None) -> bool:
        """Return False on calendar errors or when the requested slot overlaps an event."""
        if not self.service:
            return False
        try:
            start_iso, end_iso = self._calculate_start_and_end_iso(date_time_str, duration_minutes)
            start_dt = datetime.datetime.fromisoformat(start_iso)
            if start_dt <= datetime.datetime.now(datetime.timezone.utc).astimezone(start_dt.tzinfo):
                return False
            events = self.service.events().list(
                calendarId="primary", timeMin=start_iso, timeMax=end_iso,
                singleEvents=True, orderBy="startTime"
            ).execute().get("items", [])
            return not any(event.get("id") != exclude_event_id for event in events)
        except Exception as e:
            logging.error("Calendar availability check failed: %s", e, exc_info=True)
            return False

    def schedule_site_visit(self, client_name: str, phone: str, employee: str, property_title: str, date_time_str: str, notes: str = ""):
        """Creates a calendar event for a new site visit."""
        if not self.service:
            logging.error("❌ Calendar service is None. Please check token.json or credentials.")
            return {"status": "error", "message": "Calendar service unavailable"}
        try:
            if not self.is_slot_available(date_time_str):
                return {"status": "error", "message": "Requested slot is unavailable"}
            start_iso, end_iso = self._calculate_start_and_end_iso(date_time_str, duration_minutes=60)

            event_body = {
                'summary': f"Site Visit: {client_name} - {property_title}",
                'description': f"Client: {client_name}\nPhone: {phone}\nAssigned Agent: {employee}\nNotes: {notes}",
                'start': {'dateTime': start_iso, 'timeZone': 'Asia/Karachi'},
                'end': {'dateTime': end_iso, 'timeZone': 'Asia/Karachi'}
            }
            event = self.service.events().insert(calendarId='primary', body=event_body).execute()
            return {"status": "success", "eventId": event.get('id')}
        except Exception as e:
            logging.error(f"❌ Error scheduling event: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def reschedule_event(self, event_id: str, new_time_str: str):
        """Updates the start/end time of an existing calendar event."""
        if not self.service or not event_id:
            return False
        try:
            if not self.is_slot_available(new_time_str, exclude_event_id=event_id):
                return False
            start_iso, end_iso = self._calculate_start_and_end_iso(new_time_str, duration_minutes=60)
            
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            event['start'] = {'dateTime': start_iso, 'timeZone': 'Asia/Karachi'}
            event['end'] = {'dateTime': end_iso, 'timeZone': 'Asia/Karachi'}
            self.service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
            return True
        except Exception as e:
            logging.error(f"❌ Error rescheduling event {event_id}: {e}", exc_info=True)
            return False

    def cancel_event(self, event_id: str):
        """Deletes an event from Google Calendar."""
        if not self.service or not event_id:
            return False
        try:
            self.service.events().delete(calendarId='primary', eventId=event_id).execute()
            return True
        except Exception as e:
            logging.error(f"❌ Error deleting event {event_id}: {e}", exc_info=True)
            return False
