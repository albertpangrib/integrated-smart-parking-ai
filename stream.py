import cv2
import numpy as np
import easyocr
import time
import os
import logging
from flask import Flask, jsonify
from tensorflow.lite.python.interpreter import Interpreter
import firebase_admin
from firebase_admin import credentials, db

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Inisialisasi Firebase
try:
    cred = credentials.Certificate("./serviceAccountKey.json")
    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": "https://ssdesp32cam-smart-parking-default-rtdb.firebaseio.com/"
        },
    )
    logging.info("Firebase initialized successfully.")
except Exception as e:
    logging.error(f"❌ Firebase initialization failed: {str(e)}")
    exit(1)

# Load model TFLite
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

# Cek dan buat folder jika belum ada
output_dir = "captures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

app = Flask(__name__)
reader = easyocr.Reader(["en"])


def preprocess_plate(image):
    """Preprocessing gambar plat nomor sebelum OCR"""
    image_resized = cv2.resize(image, (300, 150), interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, image_bw = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_bw


@app.route("/detect/<id>", methods=["GET"])
def detect_plate(id):
    try:
        data = db.reference(f"histories/{id}/plate").get()
        if not data:
            return jsonify({"status": False, "message": "❌ Data tidak ditemukan"}), 400

        license_plate = data.get("plate", "")
        area = data.get("area", "")
        logging.info(f"License Plate: {license_plate}, Area: {area}")

        ip_address = db.reference(f"esp32cam/slot_{area}/ipAddress").get()
        logging.info(f"IP Address: {ip_address}, in Area: {area}")
        if not ip_address:
            return (
                jsonify({"status": False, "message": "❌ IP Address tidak ditemukan"}),
                400,
            )

        URL_STREAM = f"http://{ip_address}:81/stream"
        cap = cv2.VideoCapture(URL_STREAM)

        if not cap.isOpened():
            return (
                jsonify(
                    {
                        "status": False,
                        "message": "❌ Tidak dapat mengakses streaming video",
                    }
                ),
                500,
            )

        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Streaming ESP32-CAM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return (
                    jsonify(
                        {
                            "status": False,
                            "message": "❌ Streaming dihentikan oleh pengguna",
                        }
                    ),
                    400,
                )

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"status": False, "message": "❌ Gagal membaca frame"}), 500
        # Proses deteksi
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (model_width, model_height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[0]["index"])[0]

        scale_x = imW / model_width
        scale_y = imH / model_height

        detected_data = None

        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin = int(max(1, (boxes[i][0] * model_height) * scale_y))
                xmin = int(max(1, (boxes[i][1] * model_width) * scale_x))
                ymax = int(min(imH, (boxes[i][2] * model_height) * scale_y))
                xmax = int(min(imW, (boxes[i][3] * model_width) * scale_x))

                plate_image = frame[ymin:ymax, xmin:xmax]
                preprocessed_plate = preprocess_plate(plate_image)

                results_easyocr = reader.readtext(preprocessed_plate)
                extracted_text = (
                    results_easyocr[0][1].replace(" ", "").upper()
                    if results_easyocr
                    else ""
                )
                ocr_confidence = results_easyocr[0][2] if results_easyocr else 0
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{extracted_text}",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                output_path = f"captures/{id}_detected.jpg"
                cv2.imwrite(output_path, frame)
                if extracted_text:
                    logging.info(
                        f"OCR Result: {extracted_text}, OCR Confidence: {ocr_confidence}"
                    )
                else:
                    logging.warning("❌ OCR result is empty!")
                    return (
                        jsonify(
                            {
                                "status": False,
                                "message": "❌ OCR gagal membaca plat nomor",
                            }
                        ),
                        400,
                    )

                is_valid = extracted_text == license_plate

                if is_valid:

                    detected_data = {
                        "ocr_result": extracted_text,
                        "model_confidence": float(scores[i]),
                        "ocr_confidence": ocr_confidence,
                        "image_path": output_path,
                    }

                    history_ref = db.reference(f"histories/{id}")
                    history_ref.update({"status": "booked", "detected_data": detected_data})
                    logging.info("Status updated to 'booked'. Data detected saved.")

                    return jsonify(
                        {
                            "status": True,
                            "detected_plates": detected_data,
                            "message": "✅ Plat berhasil dikenali dan disimpan.",
                        }
                    )
                else:
                    logging.warning(
                        f"❌ Plate mismatch: OCR={extracted_text}, Expected={license_plate}"
                    )
                    return (
                        jsonify({"status": False, "message": "❌ Karakter tidak sesuai"}),
                        400,
                    )

    except Exception as e:
        logging.error(f"❌ Error in detect_plate: {str(e)}")
        return jsonify({"status": False, "message": f"❌ Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
