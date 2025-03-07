import os
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
from flask import Flask, request, jsonify
from tensorflow.lite.python.interpreter import Interpreter
import easyocr

# Konfigurasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = "./received_images"
RESULT_FOLDER = "./processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Path model dan label
PATH_TO_MODEL = './16000/custom_model_lite/detect.tflite'
PATH_TO_LABELS = './labelmap.txt'

# Muat label
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Muat model TensorFlow Lite
interpreter = Interpreter(model_path=PATH_TO_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_height = input_details[0]['shape'][1]
model_width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def preprocess_plate(image, filename):
    """Preprocessing pelat nomor sebelum OCR"""
    image_resized = cv2.resize(image, (300, 150), interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    _, image_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    processed_path = os.path.join(RESULT_FOLDER, filename.replace(".jpg", "_bw.jpg"))
    resized_path = os.path.join(RESULT_FOLDER, filename.replace(".jpg", "_resized.jpg"))

    cv2.imwrite(processed_path, image_bw)
    cv2.imwrite(resized_path, image_resized)
    
    return image_resized, resized_path


def extract_text_from_plate(image, filename):
    """Ekstrak teks dari pelat nomor menggunakan Tesseract dan EasyOCR"""
    processed_image, processed_path = preprocess_plate(image, filename)

    text_tesseract = pytesseract.image_to_string(processed_image, config='--psm 7').strip()

    reader = easyocr.Reader(['en'])
    results_easyocr = reader.readtext(processed_image)

    print("Tesseract OCR:", text_tesseract)
    print("EasyOCR Hasil Mentah:", results_easyocr)

    text_easyocr = results_easyocr[0][1].replace(" ", "") if results_easyocr else ""

    print("EasyOCR Cleaned:", text_easyocr)

    return text_tesseract, text_easyocr, processed_path

def tflite_detect_image(image_path, min_conf=0.5):
    """Deteksi objek dan ekstraksi teks pelat nomor"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    
    # Simpan gambar asli
    original_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path).replace(".jpg", "_original.jpg"))
    cv2.imwrite(original_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    image_resized = cv2.resize(image_rgb, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0)
    
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    
    results = []
    scale_x = imW / model_width
    scale_y = imH / model_height
    
    for i in range(len(scores)):
        if scores[i] > min_conf:
            ymin = int(max(1, (boxes[i][0] * model_height) * scale_y))
            xmin = int(max(1, (boxes[i][1] * model_width) * scale_x))
            ymax = int(min(imH, (boxes[i][2] * model_height) * scale_y))
            xmax = int(min(imW, (boxes[i][3] * model_width) * scale_x))
            object_name = labels[int(classes[i])]
            
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path).replace(".jpg", f"_{object_name}.jpg"))
            cv2.imwrite(cropped_path, cropped_image)
            
            plate_text = ""
            processed_text_path = ""
            if object_name == "license-plate":
                plate_text, plate_text_easyocr, processed_text_path = extract_text_from_plate(cropped_image, os.path.basename(image_path))

            results.append({
                "object": object_name,
                "confidence": float(scores[i]),
                "bbox": [xmin, ymin, xmax, ymax],
                "plate_text_easyocr": plate_text_easyocr,
                "plate_text_tesseract": plate_text,
                "cropped_image": cropped_path,
                "processed_image": processed_text_path
            })
            
            cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            label = f"{object_name}: {int(scores[i] * 100)}%"
            cv2.putText(image_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path).replace(".jpg", "_result.jpg"))
    cv2.imwrite(result_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    return results, {
        "original": original_path,
        "processed": result_path
    }

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    results, image_paths = tflite_detect_image(file_path)
    return jsonify({
        "detections": results,
        "image_process": image_paths
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
