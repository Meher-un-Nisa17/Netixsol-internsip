import sqlite3
import datetime
import json
import logging
import os

class CRMLoggingStore:
    """
    Centralized CRM storage layer handling call transcripts, client preferences,
    appointment lifecycle history, and follow-up reminders.
    """
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("CRM_DB_PATH", "crm_store.db")
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes relational tables for CRM data persistence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Call Transcripts Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS call_transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    transcript_text TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            """)

            # 2. Client Preferences Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_preferences (
                    phone TEXT PRIMARY KEY,
                    client_name TEXT,
                    property_type TEXT,
                    deal_type TEXT,
                    city TEXT,
                    target_area TEXT,
                    max_budget REAL,
                    min_bedrooms INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 3. Appointment History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    phone TEXT NOT NULL,
                    action_type TEXT NOT NULL, -- e.g., 'BOOKED', 'RESCHEDULED', 'CANCELLED'
                    meeting_time TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Follow-up Reminders Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS follow_up_reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT NOT NULL,
                    reminder_time TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT DEFAULT 'PENDING', -- 'PENDING', 'COMPLETED', 'DISMISSED'
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # -------------------------------------------------------------------------
    # 1. CALL TRANSCRIPTS LOGGING
    # -------------------------------------------------------------------------
    def log_transcript_turn(self, phone: str, speaker: str, text: str, session_id: str = None):
        """Logs individual conversational turns (User / Assistant / System)."""
        if not phone or not text:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO call_transcripts (phone, speaker, transcript_text, session_id)
                VALUES (?, ?, ?, ?)
            """, (phone, speaker, text, session_id))
            conn.commit()

    def get_full_transcript(self, phone: str):
        """Retrieves complete conversation history sorted chronologically."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT speaker, transcript_text, timestamp 
                FROM call_transcripts 
                WHERE phone = ? 
                ORDER BY timestamp ASC
            """, (phone,))
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # 2. CLIENT PREFERENCES
    # -------------------------------------------------------------------------
    def upsert_client_preferences(self, phone: str, state_data: dict):
        """Inserts or updates client profile preferences in real time."""
        if not phone:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO client_preferences (
                    phone, client_name, property_type, deal_type, city, target_area, max_budget, min_bedrooms, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(phone) DO UPDATE SET
                    client_name = COALESCE(EXCLUDED.client_name, client_preferences.client_name),
                    property_type = COALESCE(EXCLUDED.property_type, client_preferences.property_type),
                    deal_type = COALESCE(EXCLUDED.deal_type, client_preferences.deal_type),
                    city = COALESCE(EXCLUDED.city, client_preferences.city),
                    target_area = COALESCE(EXCLUDED.target_area, client_preferences.target_area),
                    max_budget = COALESCE(EXCLUDED.max_budget, client_preferences.max_budget),
                    min_bedrooms = COALESCE(EXCLUDED.min_bedrooms, client_preferences.min_bedrooms),
                    last_updated = CURRENT_TIMESTAMP
            """, (
                phone,
                state_data.get("client_name"),
                state_data.get("property_type"),
                state_data.get("deal_type"),
                state_data.get("city"),
                state_data.get("target_area"),
                state_data.get("max_budget"),
                state_data.get("min_bedrooms")
            ))
            conn.commit()

    def get_client_preferences(self, phone: str):
        """Retrieves stored client preferences profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM client_preferences WHERE phone = ?", (phone,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # 3. APPOINTMENT HISTORY
    # -------------------------------------------------------------------------
    def log_appointment_event(self, phone: str, action_type: str, event_id: str = None, meeting_time: str = None, notes: str = ""):
        """Logs lifecycle changes for site visits (BOOKED, RESCHEDULED, CANCELLED)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO appointment_history (event_id, phone, action_type, meeting_time, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, phone, action_type, meeting_time, notes))
            conn.commit()

    def get_appointment_history(self, phone: str):
        """Retrieves full audit trail of appointments for a specific phone number."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, action_type, meeting_time, notes, created_at 
                FROM appointment_history 
                WHERE phone = ? 
                ORDER BY created_at DESC
            """, (phone,))
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # 4. FOLLOW-UP REMINDERS
    # -------------------------------------------------------------------------
    def create_followup_reminder(self, phone: str, reminder_time: str, reason: str):
        """Creates an automated follow-up reminder for sales agents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO follow_up_reminders (phone, reminder_time, reason)
                VALUES (?, ?, ?)
            """, (phone, reminder_time, reason))
            conn.commit()

    def get_pending_reminders(self):
        """Retrieves all pending follow-up tasks due for action."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.phone, r.reminder_time, r.reason, p.client_name
                FROM follow_up_reminders r
                LEFT JOIN client_preferences p ON r.phone = p.phone
                WHERE r.status = 'PENDING'
                ORDER BY r.reminder_time ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def mark_reminder_completed(self, reminder_id: int):
        """Updates reminder state to completed once agent contacts client."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE follow_up_reminders 
                SET status = 'COMPLETED' 
                WHERE id = ?
            """, (reminder_id,))
            conn.commit()
