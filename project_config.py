# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Topics
TOPIC_SENSORS = "home/sensors/environment" # Payload: {temp, press, hum, dist}
TOPIC_VISION = "home/vision/status"       # Payload: {num_faces, brightness, recognized_name}

# Thresholds
LIDAR_THRESHOLD_MM = 1200  # 1.2 Meters (Distance to detect a person sitting)
FACE_CONFIDENCE = 0.6
