import os
import sys
import datetime

# --- CRITICAL FIXES FOR JETSON NANO ---
if "DBUS_SESSION_BUS_ADDRESS" in os.environ:
    del os.environ["DBUS_SESSION_BUS_ADDRESS"]
os.environ["SDL_VIDEO_ALLOW_SCREENSAVER"] = "1"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import json
import threading
import project_config as config
import base64
import io
import numpy as np

# --- CONFIGURATION ---
ENABLE_GPU_FRACTALS = False

# --- Global State ---
state = {
    "temp": 22.0, "hum": 0.0, "press": 0, "dist": 9999,
    "faces": 0, "brightness": 0, "user": "Unknown",
    "volume": 0,
    "cam_frame": None,
    "track": "Waiting for Mood...",
    "artist": "Spotify",
    "genre": "-",
    "album_art_data": None,
    "dominant_color": (0, 120, 215)
}

# --- Aurora Engine ---
class AuroraEngine:
    def __init__(self, width=800, height=480):
        self.w = width
        self.h = height
        self.calc_w = 160
        self.calc_h = 120
        x = np.linspace(0, 4*np.pi, self.calc_w)
        y = np.linspace(0, 4*np.pi, self.calc_h)
        self.X, self.Y = np.meshgrid(x, y)
        self.t = 0.0

    def generate(self, valence, energy, color_rgb):
        self.t += 0.1 + (energy * 0.2)
        complexity = 1.0 + ((1.0 - valence) * 2.0)
        v1 = np.sin(self.X * complexity + self.t)
        v2 = np.sin(self.Y * complexity + self.t * 0.5)
        v3 = np.sin((self.X + self.Y) * complexity * 0.5 + self.t * 0.3)
        pattern = (v1 + v2 + v3) / 3.0
        pattern = (pattern + 1.0) / 2.0
        contrast = 0.5 + (energy * 1.0)
        pattern = (pattern - 0.5) * contrast + 0.5
        pattern = np.clip(pattern, 0, 1)
        r = (pattern * color_rgb[0]).astype(np.uint8)
        g = (pattern * color_rgb[1]).astype(np.uint8)
        b = (pattern * color_rgb[2]).astype(np.uint8)
        rgb = np.dstack((r, g, b))
        pil_img = Image.fromarray(rgb)
        return pil_img.resize((self.w, self.h), Image.BICUBIC)

# --- Mood Engine ---
class MoodEngine:
    def __init__(self):
        self.valence = 0.5
        self.energy = 0.5
        self.current_mood = "Neutral"

    def calculate(self, temp, brightness, faces, volume):
        current_hour = datetime.datetime.now().hour
        dist_from_optimal = abs(temp - 22.0)
        temp_valence = max(0.0, 1.0 - (dist_from_optimal / 12.0))
        noise_penalty = 0.0
        if volume > 75: noise_penalty = (volume - 75) / 50.0 
        self.valence = max(0.0, min(1.0, temp_valence - noise_penalty))

        light_energy = brightness / 255.0
        sound_energy = volume / 100.0
        social_bonus = min(faces * 0.2, 0.4)
        time_modifier = 0.0
        if 6 <= current_hour < 11: time_modifier = 0.2 
        elif 22 <= current_hour or current_hour < 5: time_modifier = -0.3
            
        raw_energy = (light_energy * 0.4) + (sound_energy * 0.4) + social_bonus + time_modifier
        self.energy = max(0.0, min(1.0, raw_energy))
        
        if self.valence >= 0.5:
            self.current_mood = "Energetic" if self.energy >= 0.6 else "Relaxed"
        else:
            self.current_mood = "Chaotic" if self.energy >= 0.6 else "Melancholy"
        return self.valence, self.energy

aurora_engine = AuroraEngine()
mood_engine = MoodEngine()

# --- MQTT Logic ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(config.TOPIC_SENSORS)
        client.subscribe(config.TOPIC_VISION)
        client.subscribe(config.TOPIC_CAMERA)
        client.subscribe(config.TOPIC_AUDIO)
        client.subscribe(config.TOPIC_MUSIC)
        client.subscribe(config.TOPIC_CALENDAR)

def on_message(client, userdata, msg):
    global state
    try:
        if msg.topic == config.TOPIC_CAMERA:
            state["cam_frame"] = msg.payload
        elif msg.topic == config.TOPIC_MUSIC:
            data = json.loads(msg.payload.decode())
            state["track"] = data["title"]
            state["artist"] = data["artist"]
            state["genre"] = data["genres"]
            state["album_art_data"] = data["cover_data"]
            try:
                img_data = base64.b64decode(data["cover_data"])
                img = Image.open(io.BytesIO(img_data))
                rgb_img = img.convert("RGB")
                state["dominant_color"] = rgb_img.resize((1, 1)).getpixel((0, 0))
            except: 
                state["dominant_color"] = (0, 120, 215)
        elif msg.topic == config.TOPIC_CALENDAR:
            data = json.loads(msg.payload.decode())
            if data.get("user") == state["user"]:
                state["calendar"] = data.get("events", [])
        else:
            payload = json.loads(msg.payload.decode())
            if msg.topic == config.TOPIC_SENSORS:
                state["temp"] = float(payload.get("temp", state["temp"]))
                state["hum"] = float(payload.get("hum", state["hum"]))
                state["dist"] = int(payload.get("dist", state["dist"]))
            elif msg.topic == config.TOPIC_VISION:
                state["faces"] = int(payload.get("faces", 0))
                state["brightness"] = int(payload.get("brightness", 0))
                state["user"] = payload.get("user", "Unknown")
            elif msg.topic == config.TOPIC_AUDIO:
                state["volume"] = int(payload.get("volume", 0))
    except: pass

def start_mqtt(client):
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_forever()
    except: pass

# --- GUI Application ---
class DashboardApp:
    def __init__(self, root, client):
        self.root = root
        self.client = client
        self.root.title("Jetson AI Dashboard")
        
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
        self.w = self.root.winfo_screenwidth()
        self.h = self.root.winfo_screenheight()
        self.cx = self.w / 2
        self.cy = self.h / 2

        self.frame_count = 0 

        # Fonts
        self.font_xl = tkfont.Font(family="Helvetica", size=36, weight="bold")
        self.font_lg = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.font_md = tkfont.Font(family="Helvetica", size=16)
        self.font_sm = tkfont.Font(family="Helvetica", size=10)

        self.canvas = tk.Canvas(root, width=self.w, height=self.h, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Layers
        self.bg_image_id = self.canvas.create_image(0, 0, anchor="nw")
        
        # --- HEADER (Rearranged) ---
        # Mood -> Top Left
        self.txt_mood = self.canvas.create_text(30, 40, text="Mood", font=("Helvetica", 16, "bold"), fill="white", anchor="w")
        
        # Time -> Top Right
        self.txt_time = self.canvas.create_text(self.w - 30, 40, text="00:00", font=("Helvetica", 16), fill="white", anchor="e")
        
        # Sensors -> Top Right (Left of Time)
        self.txt_sensors = self.canvas.create_text(self.w - 120, 40, text="--.- C", font=("Helvetica", 16), fill="#aaaaaa", anchor="e")

        # --- LEFT SIDE: MUSIC ---
        self.art_size = 250
        self.art_placeholder = ImageTk.PhotoImage(Image.new('RGB', (self.art_size, self.art_size), color='#222222'))
        self.art_id = self.canvas.create_image(self.cx / 2, self.cy - 50, image=self.art_placeholder) 
        
        self.txt_track = self.canvas.create_text(self.cx / 2, self.cy + 100, text="Waiting for Mood...", font=("Helvetica", 18, "bold"), fill="white", width=350, justify="center")
        self.txt_artist = self.canvas.create_text(self.cx / 2, self.cy + 130, text="Spotify", font=("Helvetica", 14), fill="#aaaaaa")
        self.txt_mood_sub = self.canvas.create_text(self.cx / 2, self.cy + 160, text="Mood: Neutral", font=("Helvetica", 12), fill=self.get_hex((0,120,215)))

        # SKIP BUTTON
        btn_x = self.cx / 2
        btn_y = self.cy + 210
        self.btn_skip_bg = self.canvas.create_rectangle(btn_x - 60, btn_y - 20, btn_x + 60, btn_y + 20, fill="#333333", outline="white")
        self.btn_skip_txt = self.canvas.create_text(btn_x, btn_y, text="SKIP >>", font=("Helvetica", 12, "bold"), fill="white")
        self.canvas.tag_bind(self.btn_skip_bg, "<Button-1>", self.skip_music)
        self.canvas.tag_bind(self.btn_skip_txt, "<Button-1>", self.skip_music)

        # --- RIGHT SIDE: INFO ---
        self.txt_welcome = self.canvas.create_text(self.cx + (self.cx/2), 150, text="System Idle", font=("Helvetica", 30, "bold"), fill="white")
        
        self.agenda_items = []
        start_y = 220
        self.txt_agenda_title = self.canvas.create_text(self.cx + (self.cx/2), start_y, text="Agenda:", font=("Helvetica", 16, "bold"), fill="#aaaaaa")
        for i in range(3):
            tid = self.canvas.create_text(self.cx + (self.cx/2), start_y + 40 + (i*35), text="", font=("Helvetica", 14), fill="white")
            self.agenda_items.append(tid)

        # --- FOOTER ---
        self.cam_w, self.cam_h = 160, 120
        self.cam_id = self.canvas.create_image(self.w - 10, self.h - 10, anchor="se")

        bar_x = 20
        bar_y = self.h - 30
        self.bar_bg_id = self.canvas.create_rectangle(bar_x, bar_y, bar_x + 200, bar_y + 10, fill="#333333", outline="")
        self.bar_vol_id = self.canvas.create_rectangle(bar_x, bar_y, bar_x, bar_y + 10, fill="#00FF00", outline="")
        self.txt_debug = self.canvas.create_text(bar_x + 210, bar_y + 5, text="Vol", font=("Helvetica", 10), fill="gray", anchor="w")

        self.update_ui()

    def skip_music(self, event):
        print("Skip Clicked!")
        self.client.publish(config.TOPIC_COMMANDS, "SKIP")
        self.canvas.itemconfig(self.btn_skip_bg, fill="#555555")
        self.root.after(200, lambda: self.canvas.itemconfig(self.btn_skip_bg, fill="#333333"))

    def get_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

    def update_ui(self):
        is_present = (state["dist"] < config.LIDAR_THRESHOLD_MM) or (state["faces"] > 0)
        val, eng = mood_engine.calculate(state["temp"], state["brightness"], state["faces"], state["volume"])
        
        # Background
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            dom_color = state["dominant_color"]
            if not isinstance(dom_color, (tuple, list)) or len(dom_color) < 3:
                dom_color = (0, 120, 215)
            
            fs_fractal = aurora_engine.generate(val, eng, dom_color)
            fs_fractal = fs_fractal.resize((self.w, self.h), Image.BILINEAR)
            
            overlay = Image.new('RGBA', fs_fractal.size, (0, 0, 0, 160))
            fs_fractal = fs_fractal.convert('RGBA')
            fs_fractal = Image.alpha_composite(fs_fractal, overlay)
            
            self.fractal_photo = ImageTk.PhotoImage(fs_fractal)
            self.canvas.itemconfig(self.bg_image_id, image=self.fractal_photo)

        if self.frame_count % 10 == 0: 
            mood_payload = {"valence": val, "energy": eng}
            self.client.publish(config.TOPIC_MOOD, json.dumps(mood_payload))

        # UI Updates
        current_time = datetime.datetime.now().strftime("%H:%M")
        self.canvas.itemconfig(self.txt_time, text=current_time)
        self.canvas.itemconfig(self.txt_sensors, text=f"{state['temp']}°C | {state['hum']}%")
        
        mood_hex = self.get_hex(state["dominant_color"])
        self.canvas.itemconfig(self.txt_mood, text=f"Mood: {mood_engine.current_mood}", fill=mood_hex)
        self.canvas.itemconfig(self.txt_mood_sub, text=f"Mood: {mood_engine.current_mood}", fill=mood_hex)

        # Music Data
        self.canvas.itemconfig(self.txt_track, text=state["track"][:30])
        self.canvas.itemconfig(self.txt_artist, text=state["artist"][:30])
        
        if state["album_art_data"]:
            try:
                img_data = base64.b64decode(state["album_art_data"])
                img = Image.open(io.BytesIO(img_data)).resize((self.art_size, self.art_size))
                self.art_photo = ImageTk.PhotoImage(img)
                self.canvas.itemconfig(self.art_id, image=self.art_photo)
                state["album_art_data"] = None
            except: pass

        # Content
        if is_present:
            self.canvas.itemconfig(self.txt_welcome, text=f"Hi, {state['user']}")
            self.canvas.coords(self.txt_welcome, self.cx + (self.cx/2), 150)
            
            if state["user"] != "Unknown":
                self.canvas.itemconfig(self.txt_agenda_title, state="normal")
                events = state.get("calendar", ["Loading..."])
                if not events: events = ["No upcoming events"]
                
                for i, tid in enumerate(self.agenda_items):
                    if i < len(events):
                        self.canvas.itemconfig(tid, text=f"• {events[i]}", state="normal")
                    else:
                        self.canvas.itemconfig(tid, state="hidden")
            else:
                self.canvas.itemconfig(self.txt_agenda_title, state="hidden")
                self.canvas.itemconfig(self.agenda_items[0], text="Scanning...", state="normal")
                for i in range(1,3): self.canvas.itemconfig(self.agenda_items[i], state="hidden")
        else:
            self.canvas.itemconfig(self.txt_welcome, text="System Idle")
            self.canvas.coords(self.txt_welcome, self.cx, self.cy)
            self.canvas.itemconfig(self.txt_agenda_title, state="hidden")
            for tid in self.agenda_items:
                self.canvas.itemconfig(tid, state="hidden")

        # Camera
        if state["cam_frame"]:
            try:
                img_data = base64.b64decode(state["cam_frame"])
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((self.cam_w, self.cam_h), Image.NEAREST)
                self.cam_photo = ImageTk.PhotoImage(img)
                self.canvas.itemconfig(self.cam_id, image=self.cam_photo)
            except: pass

        # Volume
        vol_px = int((state['volume'] / 100.0) * 200)
        x1, y1, x2, y2 = self.canvas.coords(self.bar_bg_id)
        self.canvas.coords(self.bar_vol_id, x1, y1, x1 + vol_px, y2)
        self.canvas.itemconfig(self.txt_debug, text=f"Vol: {state['volume']}%")

        self.root.after(50, self.update_ui)

if __name__ == "__main__":
    client = mqtt.Client("DashboardTK")
    client.on_connect = on_connect
    client.on_message = on_message
    t = threading.Thread(target=start_mqtt, args=(client,))
    t.daemon = True
    t.start()

    if os.environ.get('DISPLAY','') == '':
        os.environ.__setitem__('DISPLAY', ':0.0')

    root = tk.Tk()
    app = DashboardApp(root, client)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
