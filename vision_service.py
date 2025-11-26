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

# --- Try Import Face Recognition ---
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[Vision] Warning: 'face_recognition' not installed.")

# --- Global State ---
current_dist = 9999
last_presence_time = 0
PROXIMITY_COOLDOWN = 5
last_face_seen_time = 0
FACE_LOCK_DURATION = 15.0 

# Recognition State
known_face_encodings = []
known_face_names = []
identified_user = "Unknown"

def load_known_faces():
    global known_face_encodings, known_face_names
    if not FACE_REC_AVAILABLE: return

    users_files = {
        "Shaunak": "shaunak.jpg",
        "Sudeep": "sudeep.jpg"
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
                else:
                    print(f"[Vision] Error: No face found in {filepath}")
            except Exception as e:
                print(f"[Vision] Error loading {filepath}: {e}")
        else:
            print(f"[Vision] Warning: {filepath} not found.")

def identify_person(rgb_image, detection):
    """
    SAFE MODE: Crops the face first, then lets dlib re-process just that crop.
    This prevents memory/coordinate errors between Jetson and Dlib.
    """
    if not FACE_REC_AVAILABLE or len(known_face_encodings) == 0:
        return "Guest"

    h, w, _ = rgb_image.shape
    
    # 1. Safe Crop Coordinates
    top = max(0, int(detection.Top))
    bottom = min(h, int(detection.Bottom))
    left = max(0, int(detection.Left))
    right = min(w, int(detection.Right))
    
    # Skip if face is too small (less than 20x20)
    if (bottom - top) < 20 or (right - left) < 20:
        return "Guest"

    # 2. Crop and Force Memory Layout
    # The .copy() is CRITICAL. It forces the array to be contiguous in memory.
    face_crop = rgb_image[top:bottom, left:right].copy()
    
    try:
        # 3. Re-Detect face in the crop (CPU)
        # We don't pass location anymore. We let dlib find it in the crop.
        # This is safer than passing external coordinates.
        crop_encodings = face_recognition.face_encodings(face_crop)
        
        if len(crop_encodings) > 0:
            unknown_encoding = crop_encodings[0]
            
            # 4. Compare
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            best_name = known_face_names[best_match_index]
            
            # Calculate Similarity % (Distance 0.0 = 100%, 1.0 = 50%)
            similarity = (1.0 - best_distance) * 100
            
            # Print Debug Info
            print(f"[Vision] Match: {best_name} | Sim: {similarity:.1f}% | Dist: {best_distance:.3f}")
            
            # Threshold: 0.6 is standard. 
            if best_distance < 0.6: 
                return best_name
            else:
                # If it's close (0.6 - 0.65), print why we rejected it
                if best_distance < 0.65:
                    print(f"[Vision] --> Close match, but rejected (Limit 0.6)")
                
    except Exception as e: 
        # If dlib crashes, print a clean error message
        print(f"[Vision] Rec Warning: {str(e)}")
    
    return "Guest"

# --- MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[Vision] MQTT Connected")
        client.subscribe(config.TOPIC_SENSORS)

def on_message(client, userdata, msg):
    global current_dist
    if msg.topic == config.TOPIC_SENSORS:
        try:
            payload = json.loads(msg.payload.decode())
            current_dist = int(payload.get("dist", 9999))
        except: pass

def main():
    global last_presence_time, last_face_seen_time, identified_user
    
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

    print("[Vision] Loading FaceNet (Detection)...")
    net = jetson.inference.detectNet("facenet-120", threshold=0.5)

    camera = None
    print("[Vision] System Ready. Waiting for Proximity...")
    
    last_ident_time = 0 

    while True:
        now = time.time()
        
        if current_dist < config.LIDAR_THRESHOLD_MM:
            last_presence_time = now
            
        is_face_locked = (now - last_face_seen_time) < FACE_LOCK_DURATION
        should_be_awake = ((now - last_presence_time) < PROXIMITY_COOLDOWN) or is_face_locked

        if should_be_awake:
            if camera is None:
                print("[Vision] Waking Camera...")
                identified_user = "Scanning..."
                last_ident_time = 0
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
                    last_face_seen_time = now
                    
                    # Run Recognition every 2 seconds
                    if (now - last_ident_time) > 2.0:
                        # Get full frame for recognition
                        cuda_mem = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format=img.format)
                        jetson.utils.cudaResize(img, cuda_mem)
                        jetson.utils.cudaDeviceSynchronize()
                        full_arr = jetson.utils.cudaToNumpy(cuda_mem)
                        
                        # Ensure Contiguous Array (Crucial for Dlib crash fix)
                        rgb_full = np.ascontiguousarray(cv2.cvtColor(full_arr, cv2.COLOR_RGBA2RGB))
                        
                        user = identify_person(rgb_full, detections[0])
                        
                        # Stickiness: Once found, stick to it until it disappears
                        if user != "Guest":
                            identified_user = user
                        # If "Guest", keep scanning (don't overwrite a known user instantly)
                        elif identified_user == "Scanning...":
                            identified_user = "Guest"
                            
                        last_ident_time = now

                # Dashboard Processing (Thumbnail)
                small_img = jetson.utils.cudaAllocMapped(width=640, height=480, format=img.format)
                jetson.utils.cudaResize(img, small_img)
                jetson.utils.cudaDeviceSynchronize()
                arr = jetson.utils.cudaToNumpy(small_img)
                brightness = int(np.mean(arr))

                # Overlay Box
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
                
            except Exception as e:
                print(f"[Vision] Loop Error: {str(e).splitlines()[0]}")

        else:
            if camera is not None:
                print("[Vision] Sleeping...")
                identified_user = "Unknown"
                try:
                    black = np.zeros((240, 320, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode('.jpg', black)
                    client.publish(config.TOPIC_CAMERA, base64.b64encode(buffer).decode('utf-8'))
                    client.publish(config.TOPIC_VISION, json.dumps({"faces": 0, "brightness": 0, "user": "Unknown"}))
                except: pass
                del camera
                camera = None
            time.sleep(0.5)

if __name__ == "__main__":
    main()
