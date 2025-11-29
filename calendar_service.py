import datetime
import os.path
import json
import time
import paho.mqtt.client as mqtt
import project_config as config

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# State
current_service = None
current_loaded_user = None
last_fetch_time = 0

def get_service_for_user(user_name):
    global current_service, current_loaded_user
    
    if user_name not in config.USER_CALENDAR_CONFIG:
        return None

    token_file = config.USER_CALENDAR_CONFIG[user_name]
    
    if current_loaded_user == user_name and current_service:
        return current_service
        
    print(f"[Calendar] Switching account to: {user_name}")

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    else:
        print(f"[Calendar] Error: {token_file} not found.")
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            print(f"[Calendar] Token refresh failed: {e}")
            return None

    try:
        service = build('calendar', 'v3', credentials=creds)
        current_service = service
        current_loaded_user = user_name
        print(f"[Calendar] Service Ready for {user_name} ✅")
        return service
    except Exception as e:
        print(f"[Calendar] Build Error: {e}")
        return None

def fetch_events(user_name):
    service = get_service_for_user(user_name)
    if not service: return []
    
    try:
        print(f"[Calendar] Querying API for {user_name}...")
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary', 
            timeMin=now,
            maxResults=3, 
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        formatted_events = []
        
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
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
        # Force reload next time
        global current_loaded_user
        current_loaded_user = None 
        return ["Calendar Error"]

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Calendar] Connected to Broker")
        client.subscribe(config.TOPIC_VISION)

def on_message(client, userdata, msg):
    global current_loaded_user, last_fetch_time
    
    try:
        payload = json.loads(msg.payload.decode())
        detected_user = payload.get("user", "Unknown")
        
        # 1. Handle "Unknown" / "Guest"
        if detected_user not in config.USER_CALENDAR_CONFIG:
            # If we were previously looking at a valid user, and now it's Unknown,
            # we MUST reset 'current_loaded_user' to None.
            # This ensures that when the valid user returns, it counts as a "Change"
            if current_loaded_user is not None:
                print(f"[Calendar] User left/lost ({current_loaded_user} -> {detected_user}). Resetting state.")
                current_loaded_user = None
                
                # Optional: Send empty clear command
                client.publish(config.TOPIC_CALENDAR, json.dumps({"user": detected_user, "events": []}))
            return

        # 2. Handle Valid User
        # Fetch if: User is different from memory OR Timer expired (5 mins)
        is_new_user = (detected_user != current_loaded_user)
        is_time_refresh = (time.time() - last_fetch_time) > 300
        
        if is_new_user or is_time_refresh:
            events = fetch_events(detected_user)
            
            out_payload = {
                "user": detected_user,
                "events": events
            }
            client.publish(config.TOPIC_CALENDAR, json.dumps(out_payload))
            
            # Update state
            last_fetch_time = time.time()
            # Note: current_loaded_user is updated inside fetch_events -> get_service_for_user
            # but we ensure it matches here to be safe
            if not current_loaded_user: 
                current_loaded_user = detected_user
                    
    except Exception as e:
        print(f"[Calendar] Msg Error: {e}")

if __name__ == "__main__":
    client = mqtt.Client("CalendarNode")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"MQTT Error: {e}")
