import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Scope required to read/write events on Google Calendar
SCOPES = ['https://www.googleapis.com/auth/calendar']

def generate_token():
    creds_path = "credentials.json"
    token_path = "token.json"

    if not os.path.exists(creds_path):
        print("❌ 'credentials.json' not found in the root directory.")
        return

    print("🔑 Opening browser for Google Calendar authorization...")
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "w") as token_file:
        token_file.write(creds.to_json())

    print("✅ 'token.json' generated successfully in the root directory!")

if __name__ == "__main__":
    generate_token()