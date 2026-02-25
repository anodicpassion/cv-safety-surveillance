import cv2
import requests
import threading
import time

VERIFY_URL = "https://6639-34-168-167-165.ngrok-free.app/verify"

# Shared variables
latest_frame = None
frame_lock = threading.Lock()
display_name = "Detecting..."
running = True


# =============================
# Thread 1: Camera Capture
# =============================
def capture_frames():
    global latest_frame, running

    cap = cv2.VideoCapture("http://192.168.252.106:4747/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffer

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Always keep only the latest frame (drop old ones)
        with frame_lock:
            latest_frame = frame

    cap.release()


# =============================
# Thread 2: Face Detection + API Call
# =============================
def process_frames():
    global latest_frame, display_name, running

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while running:
        time.sleep(0.2)  # Drop frames intentionally (process every 200ms)

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', face_crop)

            files = {
                "file": ("face.jpg", img_encoded.tobytes(), "image/jpeg")
            }

            try:
                response = requests.post(VERIFY_URL, files=files, timeout=3)
                result = response.json()

                if result.get("match"):
                    display_name = result.get("name")
                else:
                    display_name = "Unknown"

            except:
                display_name = "Error"

            break  # Only process first detected face


# =============================
# Start Threads
# =============================
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()


# =============================
# Main Display Loop
# =============================
while True:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    cv2.putText(frame,
                display_name,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

capture_thread.join()
process_thread.join()
cv2.destroyAllWindows()