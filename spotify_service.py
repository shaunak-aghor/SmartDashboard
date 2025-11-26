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

def get_mood_query(valence, energy):
    # Map Valence/Energy to Search Queries
    if energy < 0.3:
        if valence < 0.4: return "Melancholic Ambient"
        else:             return "Peaceful Piano Sleep"
    elif energy < 0.6:
        if valence < 0.4: return "Sad Indie Folk"
        elif valence < 0.7: return "Lofi Hip Hop Study"
        else:             return "Happy Acoustic Morning"
    elif energy < 0.8:
        if valence < 0.4: return "Grunge Rock 90s"
        elif valence < 0.7: return "Classic Rock Anthems"
        else:             return "Upbeat Pop Hits"
    else:
        if valence < 0.4: return "Heavy Metal Workout"
        elif valence < 0.7: return "High Tempo Techno"
        else:             return "Summer Dance Party"

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

def fetch_and_play(valence, energy):
    if not sp: return None
    
    query = get_mood_query(valence, energy)
    print(f"[Spotify] Mood: '{query}' (V:{valence:.2f} E:{energy:.2f})")
    
    try:
        # 1. Search for Playlist
        playlist_results = sp.search(q=query, type='playlist', limit=1, offset=random.randint(0, 5))
        if not playlist_results['playlists']['items']:
            playlist_results = sp.search(q=query, type='playlist', limit=1)
            
        if playlist_results['playlists']['items']:
            playlist_id = playlist_results['playlists']['items'][0]['id']
            
            # 2. Get Tracks
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
    global last_play_time, last_api_check, current_valence, current_energy
    if not sp: return

    try:
        # 1. Handle SKIP Command (Force Play)
        if msg.topic == config.TOPIC_COMMANDS:
            cmd = msg.payload.decode()
            if cmd == "SKIP":
                print("[Spotify] Skipping track...")
                data = fetch_and_play(current_valence, current_energy)
                if data:
                    client.publish(config.TOPIC_MUSIC, json.dumps(data))
                    last_play_time = time.time()

        # 2. Handle Mood Updates (Check status, play if stopped)
        elif msg.topic == config.TOPIC_MOOD:
            payload = json.loads(msg.payload.decode())
            current_valence = payload.get("valence", 0.5)
            current_energy = payload.get("energy", 0.5)
            
            # Only check Spotify status every 5 seconds to avoid rate limits
            if (time.time() - last_api_check) > 5:
                last_api_check = time.time()
                
                try:
                    # Check if music is currently playing
                    playback = sp.current_playback()
                    is_playing = playback and playback.get('is_playing')
                    
                    # Logic: If NOT playing, and we haven't tried to play recently (10s cooldown)
                    if not is_playing:
                        if (time.time() - last_play_time) > 10:
                            print("[Spotify] Silence detected. Starting new track...")
                            data = fetch_and_play(current_valence, current_energy)
                            if data:
                                client.publish(config.TOPIC_MUSIC, json.dumps(data))
                                last_play_time = time.time()
                    else:
                        # Music is playing. Do nothing. Let it finish.
                        pass
                        
                except Exception as e:
                    print(f"[Spotify] Status Check Error: {e}")
        
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
