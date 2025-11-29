import cv2
import jetson.inference
import jetson.utils
import time
import json
import numpy as np
import base64
import paho.mqtt.client as mqtt
import project_config as config
import os
import gc 

# --- Try Import Face Recognition ---
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[Vision] Warning: 'face_recognition' not installed.")

# --- Global State ---
current_dist = 9999
last_sensor_time = 0
last_presence_time = 0
PROXIMITY_COOLDOWN = 5.0 # System stays awake for 5s after Lidar triggers

# Recognition State
known_face_encodings = []
known_face_names = []
identified_user = "Unknown"

def load_known_faces():
    global known_face_encodings, known_face_names
    if not FACE_REC_AVAILABLE: return

    users_files = {
        "Shaunak": "/home/illusion/Desktop/SmartDashboard/shaunak.jpg",
        "Sudeep": "/home/illusion/Desktop/SmartDashboard/sudeep.jpg"
    }

    print("[Vision] Loading Face References...")
    for name, filepath in users_files.items():
        if os.path.exists(filepath):
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"[Vision] Loaded: {name} ✅")
            except Exception as e:
                print(f"[Vision] Error loading {filepath}: {e}")

def identify_person(rgb_image, detection):
    if not FACE_REC_AVAILABLE or len(known_face_encodings) == 0:
        return "Guest"

    h, w, _ = rgb_image.shape
    
    top = max(0, int(detection.Top))
    bottom = min(h, int(detection.Bottom))
    left = max(0, int(detection.Left))
    right = min(w, int(detection.Right))
    
    if (bottom - top) < 20 or (right - left) < 20: return "Guest"

    face_crop = rgb_image[top:bottom, left:right].copy()
    
    try:
        crop_encodings = face_recognition.face_encodings(face_crop)
        if len(crop_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, crop_encodings[0])
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            if best_distance < 0.6: 
                return known_face_names[best_match_index]
    except: pass
    
    return "Guest"

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Vision] MQTT Connected")
        client.subscribe(config.TOPIC_SENSORS)

def on_message(client, userdata, msg):
    global current_dist, last_sensor_time
    if msg.topic == config.TOPIC_SENSORS:
        try:
            payload = json.loads(msg.payload.decode())
            current_dist = int(payload.get("dist", 9999))
            last_sensor_time = time.time()
        except: pass

def main():
    global last_presence_time, identified_user, current_dist
    
    print("[Vision] Initializing AI Engine...")
    client = mqtt.Client("VisionNode")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_start()
    except:
        print("[Vision] MQTT Connection Failed")
        return

    load_known_faces()
    print("[Vision] Loading FaceNet...")
    net = jetson.inference.detectNet("facenet-120", threshold=0.5)

    camera = None
    print("[Vision] System Ready. Waiting for Proximity...")
    
    last_ident_time = 0 

    while True:
        now = time.time()
        
        if (now - last_sensor_time) > 3.0:
            current_dist = 9999

        if current_dist < config.LIDAR_THRESHOLD_MM:
            last_presence_time = now
            
        should_be_awake = (now - last_presence_time) < PROXIMITY_COOLDOWN

        if should_be_awake:
            if camera is None:
                print("[Vision] Proximity Detected! Powering Camera ON...")
                # SEND WAKE SIGNAL
                client.publish(config.TOPIC_COMMANDS, "WAKE")
                try:
                    camera = jetson.utils.videoSource("/dev/video0", argv=[
                        "--input-width=640", "--input-height=480", "--input-codec=mjpeg"
                    ])
                except:
                    time.sleep(1)
                    continue

            try:
                img = camera.Capture()
                if img is None: continue

                detections = net.Detect(img)
                num_faces = len(detections)
                
                if num_faces > 0:
                    if (now - last_ident_time) > 2.0:
                        cuda_mem = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format=img.format)
                        jetson.utils.cudaResize(img, cuda_mem)
                        jetson.utils.cudaDeviceSynchronize()
                        full_arr = jetson.utils.cudaToNumpy(cuda_mem)
                        rgb_full = np.ascontiguousarray(cv2.cvtColor(full_arr, cv2.COLOR_RGBA2RGB))
                        
                        identified_user = identify_person(rgb_full, detections[0])
                        last_ident_time = now

                # Process Thumbnail
                small_img = jetson.utils.cudaAllocMapped(width=640, height=480, format=img.format)
                jetson.utils.cudaResize(img, small_img)
                jetson.utils.cudaDeviceSynchronize()
                arr = jetson.utils.cudaToNumpy(small_img)
                brightness = int(np.mean(arr))

                bgr_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                if num_faces > 0:
                    color = (0, 255, 0) if identified_user in known_face_names else (0, 165, 255)
                    d = detections[0]
                    cv2.rectangle(bgr_img, (int(d.Left), int(d.Top)), (int(d.Right), int(d.Bottom)), color, 2)
                    cv2.putText(bgr_img, identified_user, (int(d.Left), int(d.Top)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                payload = {
                    "faces": num_faces,
                    "brightness": brightness,
                    "user": identified_user
                }
                client.publish(config.TOPIC_VISION, json.dumps(payload))
                client.publish(config.TOPIC_CAMERA, jpg_as_text)
                time.sleep(0.03)
                
            except Exception: pass

        else:
            if camera is not None:
                print("[Vision] Timeout (No Proximity). Powering Camera OFF...")
                identified_user = "Unknown"
                
                # SEND PAUSE SIGNAL
                client.publish(config.TOPIC_COMMANDS, "PAUSE")
                print("[Vision] Sent PAUSE command to Spotify.")
                
                try:
                    black = np.zeros((240, 320, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', black)
                    client.publish(config.TOPIC_CAMERA, base64.b64encode(buffer).decode('utf-8'))
                    client.publish(config.TOPIC_VISION, json.dumps({"faces": 0, "brightness": 0, "user": "Unknown"}))
                except: pass

                del camera
                camera = None
                gc.collect() 
            
            time.sleep(0.5)

if __name__ == "__main__":
    main()
