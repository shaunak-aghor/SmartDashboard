import tkinter as tk
from tkinter import font as tkfont
import paho.mqtt.client as mqtt
import json
import threading
import project_config as config
import sys
import os

# --- Global State ---
state = {
    "temp": 0, "hum": 0, "press": 0, "dist": 9999,
    "faces": 0, "brightness": 0, "user": "Unknown"
}

# --- MQTT Logic (Background Thread) ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT Connected!")
        client.subscribe(config.TOPIC_SENSORS)
        client.subscribe(config.TOPIC_VISION)
    else:
        print(f"MQTT Fail: {rc}")

def on_message(client, userdata, msg):
    global state
    try:
        payload = json.loads(msg.payload.decode())
        if msg.topic == config.TOPIC_SENSORS:
            state["temp"] = payload.get("temp", state["temp"])
            state["hum"] = payload.get("hum", state["hum"])
            state["dist"] = payload.get("dist", state["dist"])
        elif msg.topic == config.TOPIC_VISION:
            state["faces"] = payload.get("faces", 0)
            state["brightness"] = payload.get("brightness", 0)
            state["user"] = payload.get("user", "Unknown")
    except: pass

def start_mqtt():
    client = mqtt.Client("DashboardTK")
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"MQTT Error: {e}")

# --- Logic Helpers ---
def get_room_mood(temp, brightness):
    if temp > 28:
        return "Stifling" if brightness < 50 else "Energetic"
    elif temp < 20:
        return "Gloomy" if brightness < 50 else "Crisp"
    else:
        return "Cozy" if brightness < 50 else "Productive"

def get_calendar(user_name):
    return [
        "10:00 AM - Project Review",
        "12:30 PM - Lunch with Team",
        "03:00 PM - Jetson Coding"
    ]

# --- GUI Application ---
class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson AI Dashboard")
        self.root.geometry("800x480")
        self.root.configure(bg="black")
        
        # Exit on Esc
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # Styles
        self.style_bg = "black"
        self.style_fg = "white"
        self.accent = "#0078d7" # Windows Blue
        
        self.font_lg = tkfont.Font(family="Helvetica", size=30, weight="bold")
        self.font_md = tkfont.Font(family="Helvetica", size=18)
        self.font_sm = tkfont.Font(family="Helvetica", size=12)

        # --- Layout Frames ---
        # 1. Header (Always visible)
        self.frame_header = tk.Frame(root, bg=self.style_bg)
        self.frame_header.pack(fill="x", padx=20, pady=10)
        
        self.lbl_env = tk.Label(self.frame_header, text="Loading...", font=self.font_md, bg=self.style_bg, fg="gray")
        self.lbl_env.pack(side="left")

        # 2. Main Content (Visible when user present)
        self.frame_content = tk.Frame(root, bg=self.style_bg)
        self.frame_content.pack(expand=True, fill="both", padx=40)
        
        # Welcome Msg
        self.lbl_welcome = tk.Label(self.frame_content, text="", font=self.font_lg, bg=self.style_bg, fg=self.style_fg)
        self.lbl_welcome.pack(anchor="w", pady=(20, 10))
        
        # Divider
        self.divider = tk.Frame(self.frame_content, bg=self.accent, height=4)
        self.divider.pack(fill="x", pady=10)
        
        # Calendar Container
        self.frame_calendar = tk.Frame(self.frame_content, bg=self.style_bg)
        self.frame_calendar.pack(anchor="w", fill="x")
        self.lbl_cal_title = tk.Label(self.frame_calendar, text="Agenda:", font=self.font_md, bg=self.style_bg, fg=self.accent)
        self.lbl_cal_title.pack(anchor="w")
        
        self.lbl_calendar_lines = []
        for _ in range(3):
            l = tk.Label(self.frame_calendar, text="", font=self.font_sm, bg=self.style_bg, fg=self.style_fg)
            l.pack(anchor="w", pady=2)
            self.lbl_calendar_lines.append(l)

        # 3. Idle Screen (Visible when empty)
        self.frame_idle = tk.Frame(root, bg=self.style_bg)
        self.lbl_idle = tk.Label(self.frame_idle, text="System Idle", font=self.font_lg, bg=self.style_bg, fg="#333333")
        self.lbl_idle.place(relx=0.5, rely=0.5, anchor="center")

        # 4. Debug Footer
        self.lbl_debug = tk.Label(root, text="Init...", font=self.font_sm, bg=self.style_bg, fg="#555555")
        self.lbl_debug.pack(side="bottom", anchor="e", padx=10, pady=5)

        # Start Update Loop
        self.update_ui()

    def update_ui(self):
        # Check Presence Logic
        is_present = (state["dist"] < config.LIDAR_THRESHOLD_MM) or (state["faces"] > 0)
        
        # Update Header
        mood = get_room_mood(state["temp"], state["brightness"])
        env_text = f"Room: {mood}  |  {state['temp']}°C  |  {state['hum']}% RH"
        self.lbl_env.config(text=env_text)

        # Update Debug
        self.lbl_debug.config(text=f"Lidar: {state['dist']}mm | Bright: {state['brightness']}")

        # Switch Views
        if is_present:
            self.frame_idle.pack_forget() # Hide Idle
            self.frame_content.pack(expand=True, fill="both", padx=40) # Show Content
            
            # Update Content
            self.lbl_welcome.config(text=f"Welcome, {state['user']}")
            
            if state["user"] != "Unknown":
                events = get_calendar(state["user"])
                self.lbl_cal_title.config(text="Today's Agenda:")
                for i, evt in enumerate(events):
                    if i < len(self.lbl_calendar_lines):
                        self.lbl_calendar_lines[i].config(text=f"• {evt}")
            else:
                self.lbl_cal_title.config(text="Identifying User...")
                for l in self.lbl_calendar_lines: l.config(text="")
                
        else:
            self.frame_content.pack_forget() # Hide Content
            self.frame_idle.pack(expand=True, fill="both") # Show Idle

        # Schedule next update (100ms = 10 FPS)
        self.root.after(100, self.update_ui)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Start MQTT Thread
    t = threading.Thread(target=start_mqtt)
    t.daemon = True
    t.start()

    # 2. Start GUI
    # Ensure we have a display
    if os.environ.get('DISPLAY','') == '':
        print('No display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')

    root = tk.Tk()
    # Optional: Fullscreen
    # root.attributes('-fullscreen', True)
    app = DashboardApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
