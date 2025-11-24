import time
import json
import paho.mqtt.client as mqtt
import smbus2
import bme280
import VL53L0X
import project_config as config

# --- Setup Hardware ---
I2C_BUS = 1
BME_ADDR = 0x76
TOF_ADDR = 0x29

def main():
    print("[Sensors] Initializing I2C...")
    
    # Init MQTT (v1.x Style)
    client = mqtt.Client("SensorNode")
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[Sensors] MQTT Error: {e}")
        return

    # Init BME280
    try:
        bus = smbus2.SMBus(I2C_BUS)
        cal_params = bme280.load_calibration_params(bus, BME_ADDR)
        print("[Sensors] BME280 Ready")
    except Exception as e:
        print(f"[Sensors] BME280 Error: {e}")
        bus = None

    # Init VL53L0X
    try:
        tof = VL53L0X.VL53L0X(i2c_bus=I2C_BUS, i2c_address=TOF_ADDR)
        tof.open()
        tof.start_ranging(1) 
        print("[Sensors] VL53L0X Ready")
    except Exception as e:
        print(f"[Sensors] VL53L0X Error: {e}")
        tof = None

    print("[Sensors] Publishing data...")

    try:
        while True:
            payload = {
                "temp": 0, "hum": 0, "press": 0, "dist": 8190
            }

            # Read BME
            if bus:
                try:
                    data = bme280.sample(bus, BME_ADDR, cal_params)
                    payload["temp"] = round(data.temperature, 1)
                    payload["hum"] = round(data.humidity, 1)
                    payload["press"] = round(data.pressure, 0)
                except: pass

            # Read Lidar
            if tof:
                try:
                    dist = tof.get_distance()
                    if dist > 0:
                        payload["dist"] = dist
                except: pass

            # Publish
            client.publish(config.TOPIC_SENSORS, json.dumps(payload))
            
            time.sleep(0.5) 

    except KeyboardInterrupt:
        if tof: tof.stop_ranging()
        if bus: bus.close()
        client.loop_stop()

if __name__ == "__main__":
    main()
