import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.calendar_integration import GoogleCalendarManager

cal = GoogleCalendarManager(token_path="token.json")

print("--- Testing Calendar Directly ---")
res = cal.schedule_site_visit(
    client_name="Test Client",
    phone="03219876543",
    employee="Test Agent",
    property_title="Lahore 1 Kanal House",
    date_time_str="2026-10-20 14:00"
)

print("Result:", res)