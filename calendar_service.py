import datetime
import os.path
import json
import time
import paho.mqtt.client as mqtt
import project_config as config

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

creds = None
service = None
last_user = None
last_fetch_time = 0

def setup_google_calendar():
    global creds, service
    print("[Calendar] Authenticating with Google...")
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except:
                print("[Calendar] Token expired and refresh failed.")
                return False
        else:
            if not os.path.exists('credentials.json'):
                print("[Calendar] ERROR: credentials.json not found!")
                return False
                
            # This will open a browser window for login
            # On headless Jetson, you might need to run this once on a PC 
            # and copy the 'token.json' file over.
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('calendar', 'v3', credentials=creds)
        print("[Calendar] Service Ready ✅")
        return True
    except Exception as e:
        print(f"[Calendar] Build Error: {e}")
        return False

def fetch_events(calendar_id):
    if not service: return []
    
    try:
        print(f"[Calendar] Fetching for ID: {calendar_id}")
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        
        events_result = service.events().list(
            calendarId=calendar_id, 
            timeMin=now,
            maxResults=3, 
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        formatted_events = []
        
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            # Parse ISO format to simple time (e.g., "10:00 AM")
            try:
                dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                time_str = dt.strftime("%I:%M %p")
            except:
                time_str = "All Day"
                
            summary = event.get('summary', 'Busy')
            formatted_events.append(f"{time_str} - {summary}")
            
        if not formatted_events:
            formatted_events = ["No upcoming events"]
            
        return formatted_events
        
    except Exception as e:
        print(f"[Calendar] API Error: {e}")
        return ["Error fetching calendar"]

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Calendar] Connected to Broker")
        client.subscribe(config.TOPIC_VISION)

def on_message(client, userdata, msg):
    global last_user, last_fetch_time
    
    try:
        payload = json.loads(msg.payload.decode())
        current_user = payload.get("user", "Unknown")
        
        # Only fetch if user changed OR it's been 5 minutes
        is_new_user = (current_user != last_user) and (current_user != "Unknown")
        is_time_refresh = (time.time() - last_fetch_time) > 300
        
        if is_new_user or (is_time_refresh and current_user != "Unknown"):
            
            # Check if we have a map for this user
            if current_user in config.CALENDAR_MAP:
                cal_id = config.CALENDAR_MAP[current_user]
                events = fetch_events(cal_id)
                
                # Publish result
                out_payload = {
                    "user": current_user,
                    "events": events
                }
                client.publish(config.TOPIC_CALENDAR, json.dumps(out_payload))
                print(f"[Calendar] Sent events for {current_user}")
                
                last_user = current_user
                last_fetch_time = time.time()
            else:
                # User recognized visually, but no calendar config
                # Don't spam updates
                if is_new_user:
                    print(f"[Calendar] No calendar mapped for {current_user}")
                    
    except Exception as e:
        print(f"[Calendar] Msg Error: {e}")

if __name__ == "__main__":
    if setup_google_calendar():
        client = mqtt.Client("CalendarNode")
        client.on_connect = on_connect
        client.on_message = on_message
        
        try:
            client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"MQTT Error: {e}")
    else:
        print("[Calendar] Failed to init Google API. Check credentials.json.")
