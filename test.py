import cv2
import numpy as np
import easyocr
import time
import os
import logging
from flask import Flask, jsonify, send_from_directory
from tensorflow.lite.python.interpreter import Interpreter
import firebase_admin
from firebase_admin import credentials, db

logging.basicConfig(level=logging.INFO)

TEMP_FOLDER = "tmp"

try:
    cred = credentials.Certificate("./serviceAccountKey.json")
    firebase_admin.initialize_app(
        cred, {"databaseURL": "https://ssdesp32cam-smart-parking-default-rtdb.firebaseio.com/"}
    )
    logging.info("Firebase initialized successfully.")
except Exception as e:
    logging.error(f"❌ Firebase initialization failed: {str(e)}")
    exit(1)

PATH_TO_MODEL = "./16000/custom_model_lite/detect.tflite"
PATH_TO_LABELS = "./labelmap.txt"

try:
    with open(PATH_TO_LABELS, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    interpreter = Interpreter(model_path=PATH_TO_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_height = input_details[0]["shape"][1]
    model_width = input_details[0]["shape"][2]
    float_input = input_details[0]["dtype"] == np.float32
    
    input_mean = 127.5
    input_std = 127.5
    logging.info("Model TFLite loaded successfully.")
except Exception as e:
    logging.error(f"❌ Model loading failed: {str(e)}")
    exit(1)

app = Flask(__name__)
reader = easyocr.Reader(["en"])

def preprocess_plate(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, image_bw = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_bw
  
@app.route('/tmp/<path:filename>', methods=['GET'])
  
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, TEMP_FOLDER), filename)

@app.route("/detect/<id>", methods=["GET"])
def detect_plate(id):
    try:
        data = db.reference(f"histories/{id}/").get()
        if not data:
            return jsonify({"status": False, "message": "❌ Data tidak ditemukan"}), 400
        history_id = data.get("id", "")
        print(history_id)
        license_plate = data.get("plate", {}).get("plate", "").upper()
        print(license_plate)
        area = data.get("plate", {}).get("area", "")
        print(area)
        ip_address = db.reference(f"esp32cam/slot_{area}/ipAddress").get()
        if not ip_address:
            return jsonify({"status": False, "message": "❌ IP Address tidak ditemukan"}), 400

        URL_STREAM = f"http://{ip_address}:81/stream"
        cap = cv2.VideoCapture(URL_STREAM)
        if not cap.isOpened():
            return jsonify({"status": False, "message": "❌ Tidak dapat mengakses streaming video"}), 500
        
        detected_texts = []
        matched = False
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(frame, (model_width, model_height))
            input_data = np.expand_dims(image_resized, axis=0)
            if float_input:
                input_data = (np.float32(input_data) - input_mean) / input_std
            
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            
            boxes = interpreter.get_tensor(output_details[1]["index"])[0]
            scores = interpreter.get_tensor(output_details[0]["index"])[0]
            
            for i in range(len(scores)):
                if scores[i] > 0.5:
                    ymin = int(boxes[i][0] * imH)
                    xmin = int(boxes[i][1] * imW)
                    ymax = int(boxes[i][2] * imH)
                    xmax = int(boxes[i][3] * imW)
                    
                    plate_image = frame[ymin:ymax, xmin:xmax]
                    preprocessed_plate = preprocess_plate(plate_image)
                    results_easyocr = reader.readtext(preprocessed_plate)
                    detected_text = "".join([res[1].replace(" ", "").upper() for res in results_easyocr])
                    
                    if detected_text:
                        detected_texts.append(detected_text)
                        logging.info(f"OCR Result: {detected_text}")
                    if results_easyocr:
                        ocr_confidence = results_easyocr[0][2]
                    model_confidence = float(scores[i])
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{model_confidence}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, detected_text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if detected_text == license_plate:
                        timestamp_id = str(int(time.time()))
                        output_path = f"tmp/{timestamp_id}_detected.jpg"
                        cv2.imwrite(output_path, frame)
                        matched = True
                        logging.info("✅ License plate matched! Stopping stream.")
                        break
            
            cv2.imshow("License Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q") or matched:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if matched:
            results= {
              "detected_texts": detected_texts[-1],
                "model_confidence": model_confidence,
                "ocr_confidence": ocr_confidence,
                "output_path": output_path
            }
            db.reference(f"histories/{id}").update({
                "status": "booked",
                "result": results
                
            })
            return jsonify({
                "status": True,
                "message": "✅ Plat berhasil dikenali dan disimpan.",
                "result": results,
                "historyId": history_id,
            }), 200
        else:
            return jsonify({"status": False, "message": "❌ Plat tidak ditemukan atau tidak cocok.", "detected_texts": detected_texts})
    except Exception as e:
        logging.error(f"❌ Error in detect_plate: {str(e)}")
        return jsonify({"status": False, "message": f"❌ Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
