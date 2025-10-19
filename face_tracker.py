import cv2
import os
import time

# SETUP

output_folder = "faces"
os.makedirs(output_folder, exist_ok=True)

# Load DNN face detector (included in OpenCV)
prototxt_path = "dnn/deploy.prototxt"
model_path = "dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"

if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
    print("[ERROR] DNN model files not found. Make sure they are inside the 'dnn' folder.")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(0)

last_faces = []
last_detect_time = 0

# Timestamp overlay

def add_timestamp(frame):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]
    x = 10
    y = frame.shape[0] - 10
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                  (x + text_size[0] + 5, y + 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, timestamp, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

# Check for new faces

def is_new_face(face, previous_faces, threshold=50):
    x, y, w, h = face
    for (px, py, pw, ph) in previous_faces:
        if abs(x - px) < threshold and abs(y - py) < threshold:
            return False
    return True

# MAIN LOOP

print("[INFO] Improved face tracker started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Create a blob for DNN detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    current_faces = []
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # filter weak detections
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            current_faces.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Only save if enough time has passed to avoid spam
    if time.time() - last_detect_time > 1:
        for (x, y, fw, fh) in current_faces:
            if is_new_face((x, y, fw, fh), last_faces):
                save_frame = frame.copy()
                save_frame = add_timestamp(save_frame)
                timestamp_file = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(output_folder, f"face_{timestamp_file}.png")
                cv2.imwrite(filename, save_frame)
                print(f"[+] New face detected (confident) — saved as {filename}")
                last_detect_time = time.time()

    last_faces = current_faces
    cv2.imshow("Improved Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
