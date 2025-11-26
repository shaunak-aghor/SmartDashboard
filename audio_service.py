import pyaudio
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import project_config as config

# --- Config ---
# Reduced chunk size for faster updates (lower latency)
# 1024 samples @ 44100Hz = ~23ms latency (approx 43 FPS)
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# --- Dynamic Gain State ---
global_max_peak = 500 
decay_factor = 0.995 

def get_webcam_mic_index(p):
    count = p.get_device_count()
    print(f"[Audio] Scanning {count} audio devices...")
    target_index = None
    for i in range(count):
        try:
            info = p.get_device_info_by_index(i)
            name = info.get('name', '')
            max_inputs = info.get('maxInputChannels', 0)
            if max_inputs > 0:
                if target_index is None:
                    if any(x in name for x in ["USB", "Webcam", "Camera", "C920", "C922", "Mic", "UAC"]):
                        print(f"[Audio] ✅ Selected: '{name}' at Index {i}")
                        target_index = i
        except: pass
    
    if target_index is None: 
        print("[Audio] ⚠️ No specific Webcam found. Will use System Default.")
    return target_index

def main():
    global global_max_peak
    print("[Audio] Initializing Microphone...")
    
    # 1. MQTT
    client = mqtt.Client("AudioNode")
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[Audio] MQTT Error: {e}")
        return

    # 2. Audio Setup
    p = pyaudio.PyAudio()
    stream = None
    
    try:
        mic_index = get_webcam_mic_index(p)
        try:
            print(f"[Audio] Attempting to open device {mic_index} at {RATE}Hz...")
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            input_device_index=mic_index, frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"[Audio] Failed to open specific device ({e}). Trying Default Device at 48000Hz...")
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=48000, input=True,
                            frames_per_buffer=CHUNK)
                            
    except Exception as e:
        print(f"[Audio] CRITICAL ERROR: Could not open ANY microphone: {e}")
        return

    print("[Audio] Listening... (Check terminal for RMS values)")
    last_print = time.time()

    try:
        while True:
            try:
                # Read raw data (Blocking call - acts as natural timer)
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # 1. Calculate Raw Loudness (RMS)
                rms = np.sqrt(np.mean(audio_data.astype(np.int64)**2))
                
                # 2. Auto-Update Peak
                if rms > global_max_peak:
                    global_max_peak = rms
                else:
                    global_max_peak *= decay_factor
                    if global_max_peak < 300: global_max_peak = 300 

                # 3. Calculate Percentage
                if rms < 5: 
                    volume_norm = 0
                else:
                    volume_norm = int((rms / global_max_peak) * 100)
                    volume_norm = min(100, int(volume_norm * 1.5)) 

                # Publish immediately (No extra sleep!)
                client.publish(config.TOPIC_AUDIO, json.dumps({"volume": volume_norm}))
                
                # Debug Print (Every 2 seconds to reduce console spam)
                if time.time() - last_print > 2:
                    print(f"[Audio] RMS: {int(rms)} | Peak: {int(global_max_peak)} | Vol: {volume_norm}%")
                    last_print = time.time()
                
            except IOError as e:
                # Only sleep if there is an error to prevent CPU spinning
                print(f"[Audio] Read Error: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
