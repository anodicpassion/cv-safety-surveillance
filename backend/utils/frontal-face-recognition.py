import face_recognition
import cv2
import os
import sys

KNOWN_DIR = "../known"

known_encodings = []
known_names = []

# Load known faces
if not os.path.exists(KNOWN_DIR):
    print(f"Known faces directory not found: {KNOWN_DIR}")
    sys.exit(1)

for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_DIR, filename)
        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"⚠️ No face found in {filename}, skipping")
            continue

        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"Loaded {filename}")

print(f"\nLoaded {len(known_names)} known faces\n")

# Start webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Could not open webcam")
    sys.exit(1)

print("🎥 Webcam started — press 'q' to quit")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]

    # Detect and encode
    locations = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, locations)

    for (top, right, bottom, left), encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=0.5
        )

        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            1,
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
