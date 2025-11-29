import spotipy
from spotipy.oauth2 import SpotifyOAuth
import paho.mqtt.client as mqtt
import json
import time
import requests
import base64
import random
import project_config as config

# --- Spotify Setup ---
sp = None

def setup_spotify():
    global sp
    print("[Spotify] Authenticating (User Mode)...")
    try:
        if "YOUR_CLIENT_ID" in config.SPOTIFY_CLIENT_ID:
            print("[Spotify] ERROR: Keys not set in project_config.py")
            return False

        auth_manager = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=config.SPOTIFY_SCOPE,
            open_browser=False 
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        user = sp.current_user()
        print(f"[Spotify] Logged in as: {user['display_name']}")
        return True
        
    except Exception as e:
        print(f"[Spotify] Auth Failed: {e}")
        sp = None
        return False

# --- State ---
last_play_time = 0
last_api_check = 0
current_valence = 0.5
current_energy = 0.5
system_status = "IDLE" # Start as IDLE so music doesn't play until someone is detected

def get_mood_query(valence, energy):
    if energy < 0.3:
        if valence < 0.4: return "Dark Ambient Instrumental"     
        else:             return "Peaceful Piano Instrumental"   
    elif energy < 0.6:
        if valence < 0.4: return "Sad Cello Instrumental"
        elif valence < 0.7: return "Lofi Beats Instrumental"     
        else:             return "Acoustic Guitar Instrumental"
    elif energy < 0.8:
        if valence < 0.4: return "Post-Rock Instrumental" 
        elif valence < 0.7: return "Synthwave Instrumental"      
        else:             return "Upbeat Jazz Instrumental"
    else:
        if valence < 0.4: return "Dark Techno Instrumental"      
        elif valence < 0.7: return "Drum and Bass Instrumental"  
        else:             return "Deep House Instrumental" 

def play_track(uri):
    try:
        devices = sp.devices()
        active_device = None
        for d in devices['devices']:
            if d['is_active']:
                active_device = d['id']
                break
        
        if not active_device and devices['devices']:
            active_device = devices['devices'][0]['id']

        if active_device:
            sp.start_playback(device_id=active_device, uris=[uri])
            print("[Spotify] ▶️ Playback Started")
        else:
            print("[Spotify] ⚠️ No active device found! Open Web Player.")
            
    except Exception as e:
        print(f"[Spotify] Playback Error: {e}")

def pause_track():
    try:
        current = sp.current_playback()
        if current and current.get('is_playing'):
            sp.pause_playback()
            print("[Spotify] ⏸️ Playback Paused (System Idle)")
    except Exception as e:
        print(f"[Spotify] Pause Error: {e}")

def fetch_and_play(valence, energy):
    if not sp: return None
    
    query = get_mood_query(valence, energy)
    print(f"[Spotify] Mood: '{query}' (V:{valence:.2f} E:{energy:.2f})")
    
    try:
        playlist_results = sp.search(q=query, type='playlist', limit=1, offset=random.randint(0, 5))
        if not playlist_results['playlists']['items']:
            playlist_results = sp.search(q=query, type='playlist', limit=1)
            
        if playlist_results['playlists']['items']:
            playlist_id = playlist_results['playlists']['items'][0]['id']
            track_results = sp.playlist_items(playlist_id, limit=20, offset=random.randint(0, 5))
            items = track_results['items']
            
            if items:
                item = random.choice(items)
                track = item.get('track')
                if not track: return None
                
                name = track['name']
                artist = track['artists'][0]['name']
                uri = track['uri']
                album_url = track['album']['images'][0]['url']
                
                play_track(uri)
                
                img_resp = requests.get(album_url, timeout=5)
                img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
                
                return {
                    "title": name,
                    "artist": artist,
                    "cover_data": img_b64,
                    "genres": query 
                }
    except Exception as e:
        print(f"[Spotify] Error: {e}")
    return None

# --- MQTT Logic ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Spotify] Connected to Broker")
        client.subscribe(config.TOPIC_MOOD)
        client.subscribe(config.TOPIC_COMMANDS)

def on_message(client, userdata, msg):
    global last_play_time, last_api_check, current_valence, current_energy, system_status
    if not sp: return

    try:
        # 1. Handle Commands
        if msg.topic == config.TOPIC_COMMANDS:
            cmd = msg.payload.decode()
            
            if cmd == "SKIP":
                print("[Spotify] Skipping track...")
                data = fetch_and_play(current_valence, current_energy)
                if data:
                    client.publish(config.TOPIC_MUSIC, json.dumps(data))
                    last_play_time = time.time()
            
            elif cmd == "PAUSE":
                system_status = "IDLE"
                pause_track()
                
            elif cmd == "WAKE":
                system_status = "ACTIVE"
                print("[Spotify] System Woke Up. Ready to play.")

        # 2. Handle Mood Updates
        elif msg.topic == config.TOPIC_MOOD:
            payload = json.loads(msg.payload.decode())
            current_valence = payload.get("valence", 0.5)
            current_energy = payload.get("energy", 0.5)
            
            # Only check status if System is ACTIVE
            if system_status == "ACTIVE":
                if (time.time() - last_api_check) > 5:
                    last_api_check = time.time()
                    try:
                        playback = sp.current_playback()
                        is_playing = playback and playback.get('is_playing')
                        
                        # Only play if stopped AND we are active
                        if not is_playing:
                            if (time.time() - last_play_time) > 10:
                                print("[Spotify] Silence detected & Active. Playing...")
                                data = fetch_and_play(current_valence, current_energy)
                                if data:
                                    client.publish(config.TOPIC_MUSIC, json.dumps(data))
                                    last_play_time = time.time()
                    except: pass
            
    except Exception as e:
        print(f"[Spotify] Msg Error: {e}")

if __name__ == "__main__":
    if setup_spotify():
        client = mqtt.Client("SpotifyNode")
        client.on_connect = on_connect
        client.on_message = on_message
        
        try:
            client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"MQTT Connection Error: {e}")
    else:
        print("[Spotify] Init Failed.")
