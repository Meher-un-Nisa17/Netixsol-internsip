import os
import sqlite3
import logging

class VisitDatabaseManager:
    """SQLite Database Manager to persist and manage client site visit records."""
    
    def __init__(self, db_file="data/visits.db"):
        os.makedirs(os.path.dirname(db_file), exist_ok=True) if os.path.dirname(db_file) else None
        self.db_file = db_file
        self._initialize_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self):
        """Creates the visits table if it does not exist."""
        with self._get_connection() as conn:
            existing = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='visits'").fetchone()
            if existing and "PHONE TEXT UNIQUE" in (existing[0] or "").upper():
                conn.execute("ALTER TABLE visits RENAME TO visits_legacy")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_name TEXT,
                    phone TEXT NOT NULL,
                    property_title TEXT,
                    city TEXT,
                    target_area TEXT,
                    meeting_time TEXT,
                    employee_name TEXT,
                    employee_email TEXT,
                    event_id TEXT
                )
            """)
            legacy = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='visits_legacy'").fetchone()
            if legacy:
                conn.execute("""INSERT INTO visits (client_name, phone, property_title, city, target_area,
                    meeting_time, employee_name, employee_email, event_id)
                    SELECT client_name, phone, property_title, city, target_area, meeting_time,
                    employee_name, employee_email, event_id FROM visits_legacy""")
                conn.execute("DROP TABLE visits_legacy")
            conn.commit()

    def save_visit(self, visit_data: dict):
        """Creates a booking, or updates the matching calendar event."""
        emp = visit_data.get("assigned_employee") or {}
        with self._get_connection() as conn:
            values = (
                visit_data.get("client_name"),
                visit_data.get("phone"),
                visit_data.get("property_title"),
                visit_data.get("city"),
                visit_data.get("target_area"),
                visit_data.get("meeting_time"),
                emp.get("name"),
                emp.get("email"),
                visit_data.get("event_id")
            )
            if visit_data.get("event_id"):
                result = conn.execute("""UPDATE visits SET client_name=?, phone=?, property_title=?, city=?, target_area=?,
                    meeting_time=?, employee_name=?, employee_email=? WHERE event_id=?""", values[:8] + (values[8],))
            else:
                result = None
            if not result or result.rowcount == 0:
                conn.execute("""INSERT INTO visits (client_name, phone, property_title, city, target_area,
                    meeting_time, employee_name, employee_email, event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", values)
            conn.commit()

    def find_visit_by_phone(self, phone: str) -> dict:
        """Fetches a saved visit record by phone number."""
        if not phone:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM visits WHERE phone = ? ORDER BY id DESC LIMIT 1", (str(phone).strip(),))
            row = cursor.fetchone()
            if row:
                return {
                    "client_name": row["client_name"],
                    "phone": row["phone"],
                    "property_title": row["property_title"],
                    "city": row["city"],
                    "target_area": row["target_area"],
                    "meeting_time": row["meeting_time"],
                    "assigned_employee": {
                        "name": row["employee_name"],
                        "email": row["employee_email"]
                    },
                    "event_id": row["event_id"]
                }
        return None

    def remove_visit(self, phone: str, event_id: str = None):
        """Deletes one booking by event id, or the latest booking for the phone."""
        if not phone:
            return
        with self._get_connection() as conn:
            if event_id:
                conn.execute("DELETE FROM visits WHERE phone = ? AND event_id = ?", (str(phone).strip(), event_id))
            else:
                conn.execute("DELETE FROM visits WHERE id = (SELECT id FROM visits WHERE phone = ? ORDER BY id DESC LIMIT 1)", (str(phone).strip(),))
            conn.commit()
