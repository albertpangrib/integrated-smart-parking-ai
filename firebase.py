import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./serviceAccountKey.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ssdesp32cam-smart-parking-default-rtdb.firebaseio.com/'
})

HISTORY_ID = '-OKRA1-8anY7QsQGZuD8'

data = db.reference(f'histories/{HISTORY_ID}/plate').get()

if data:
    license_plate = data.get("plate")
    area = data.get("area")
else:
    license_plate, area = None, None

ip_address = None
if area is not None:
    ip_address = db.reference(f'esp32cam/slot_{area}/ipAddress').get()

print(f"License Plate: {license_plate}")
print(f"Area: {area}")
print(f"IP Address: {ip_address}")

extracted_text = "BK1403EOS"
output_path = "1234.jpg"
ocr_confidence = 90
scores = [[99], [12]]

model_confidence = float(scores[0][0])

detected_data = {
    "hasil": extracted_text,
    "model_confidence": model_confidence,
    "ocr_confidence": ocr_confidence,
    "image_path": output_path
}

is_valid = license_plate is not None and extracted_text == license_plate

history_ref = db.reference(f'histories/{HISTORY_ID}')
if is_valid:
    history_ref.update({
        "status": "booked",
        "detected_data": detected_data
    })
    print("Status updated to 'booked'. Data detected saved.")
else:
    print("License plate does not match. No update performed.")
