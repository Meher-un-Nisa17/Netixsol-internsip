import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils.settings import BUSINESS_EMAIL

class EmailAutomationManager:
    """Plain-text Email Manager for automated CRM notifications."""
    
    def __init__(self):
        self.notification_email = BUSINESS_EMAIL
        self.sender_email = os.environ.get("SENDER_EMAIL", "") or self.notification_email
        self.sender_password = os.environ.get("SENDER_PASSWORD", "")
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_visit_notification(self, visit_info: dict):
        """Sends booking confirmation to assigned employee."""
        emp = visit_info.get("assigned_employee") or {}
        receiver_email = self.notification_email
        
        subject = f"New Site Visit Booked: {visit_info.get('client_name', 'Client')} - {visit_info.get('property_title', 'Property')}"
        body = f"""Hello {emp.get('name', 'Agent')},

A new property site visit has been successfully scheduled. Here are the complete details:

- Client Name: {visit_info.get('client_name')}
- Phone Number: {visit_info.get('phone')}
- Property Title: {visit_info.get('property_title')}
- Location: {visit_info.get('target_area', 'N/A')}, {visit_info.get('city', 'N/A')}
- Visit Date & Time: {visit_info.get('meeting_time')}
- Assigned Employee: {emp.get('name')} ({emp.get('email')})

Real Estate Hub Automated CRM Notification
"""
        self._send_mail(receiver_email, subject, body)

    def send_cancellation_notification(self, visit_info: dict):
        """Sends cancellation alert to assigned employee."""
        emp = visit_info.get("assigned_employee") or {}
        receiver_email = self.notification_email
        
        subject = f"SITE VISIT CANCELED: {visit_info.get('client_name', 'Client')} - {visit_info.get('property_title', 'Property')}"
        body = f"""Hello {emp.get('name', 'Agent')},

The scheduled property site visit has been CANCELED by the client. Below were the booked details:

- Client Name: {visit_info.get('client_name')}
- Phone Number: {visit_info.get('phone')}
- Property: {visit_info.get('property_title')}
- Location: {visit_info.get('target_area', 'N/A')}, {visit_info.get('city', 'N/A')}
- Was Scheduled For: {visit_info.get('meeting_time')}
- Event ID: {visit_info.get('event_id')}

Real Estate Hub CRM Automation
"""
        self._send_mail(receiver_email, subject, body)

    def send_reschedule_notification(self, visit_info: dict, new_time_str: str):
        """Sends rescheduling alert to assigned employee."""
        emp = visit_info.get("assigned_employee") or {}
        receiver_email = self.notification_email
        
        subject = f"SITE VISIT RESCHEDULED: {visit_info.get('client_name', 'Client')}"
        body = f"""Hello {emp.get('name', 'Agent')},

A site visit has been RESCHEDULED to a new time. Updated details:

- Client Name: {visit_info.get('client_name')}
- Phone Number: {visit_info.get('phone')}
- Property: {visit_info.get('property_title')}
- New Date & Time: {new_time_str} (Previous: {visit_info.get('meeting_time')})

Real Estate Hub CRM Automation
"""
        self._send_mail(receiver_email, subject, body)

    def _send_mail(self, to_email: str, subject: str, body: str):
        if not self.sender_email or not self.sender_password:
            logging.warning("Email credentials missing in environment variables.")
            return
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = self.notification_email
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.notification_email, msg.as_string())
            logging.info(f"📧 Plain Text Email Sent Successfully to {to_email}")
        except Exception as e:
            logging.error(f"❌ Email Error: {e}")
