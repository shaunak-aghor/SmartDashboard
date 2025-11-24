import jetson.inference
import jetson.utils
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
import project_config as config

def main():
    print("[Vision] Initializing GPU & Camera...")

    # Init MQTT (v1.x Style)
    client = mqtt.Client("VisionNode")
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_start()
    except:
        print("[Vision] MQTT Connection Failed")
        return

    # Init Camera (Force MJPEG for stability)
    # Using /dev/video0 for USB Camera
    camera = jetson.utils.videoSource("/dev/video0", argv=[
        "--input-width=640", "--input-height=480", "--input-codec=mjpeg"
    ])

    # Init Face Detection
    # Using 'facenet-120' which you downloaded earlier
    net = jetson.inference.detectNet("facenet-120", threshold=0.5)

    print("[Vision] Stream starting...")

    while True:
        # 1. Capture
        img = camera.Capture()
        if img is None: continue

        # 2. Detect Faces
        detections = net.Detect(img)
        num_faces = len(detections)
        
        # 3. Calculate Brightness
        # Map GPU image to CPU to do numpy math
        small_img = jetson.utils.cudaAllocMapped(width=64, height=48, format=img.format)
        jetson.utils.cudaResize(img, small_img)
        jetson.utils.cudaDeviceSynchronize()
        
        arr = jetson.utils.cudaToNumpy(small_img)
        brightness = int(np.mean(arr))

        # 4. Identify
        user_name = "Unknown"
        if num_faces > 0:
            # Simple proximity check: if face is big (close), assume it's Admin
            if detections[0].Area > 50000: 
                user_name = "Admin User"

        # 5. Publish
        payload = {
            "faces": num_faces,
            "brightness": brightness,
            "user": user_name
        }
        
        client.publish(config.TOPIC_VISION, json.dumps(payload))
        time.sleep(0.1)

if __name__ == "__main__":
    main()
