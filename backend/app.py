import os
import csv
import threading
import time
import sqlite3
import json
import requests
from datetime import datetime
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# =====================================
#  CONFIGURATION
# =====================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE_DIR, "ppe_monitor.db")
MODEL_PATH   = "models/ppe.pt"
CONTACTS_CSV = os.path.join(BASE_DIR, "contacts.csv")
KNOWN_DIR    = os.path.join(BASE_DIR, "known")
VERIFY_URL   = "https://7cbb-34-171-105-180.ngrok-free.app/verify"
SMS_URL      = "https://www.fast2sms.com/dev/bulkV2"
SMS_AUTH     = "FAST_2_SMS_API_KEY"

PPE_CLASSES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

os.makedirs(KNOWN_DIR, exist_ok=True)

# =====================================
#  CONTACTS
# =====================================
# FIX: Use RLock (reentrant lock) so the same thread can acquire it multiple
#      times without deadlocking (e.g. reload_contacts called inside a block
#      that already holds the lock).
contacts_lock = threading.RLock()   # ← was threading.Lock()

def load_contacts(csv_path: str):
    admin_number = None
    workers = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role   = row.get("role",   "").strip().lower()
                name   = row.get("name",   "").strip()
                number = row.get("number", "").strip()
                if not name or not number:
                    continue
                if role == "admin" and admin_number is None:
                    admin_number = number
                    print(f"[CONTACTS] Admin: {name} -> {number}")
                elif role == "worker":
                    workers[name.lower()] = number
                    print(f"[CONTACTS] Worker: {name} -> {number}")
    except FileNotFoundError:
        print(f"[CONTACTS] WARNING: {csv_path} not found.")
    return admin_number, workers

def _read_all_rows(csv_path: str):
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ["role", "name", "number"]
            rows = list(reader)
            return fieldnames, rows
    except FileNotFoundError:
        return ["role", "name", "number"], []

def _write_all_rows(csv_path: str, fieldnames, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

ADMIN_NUMBER, WORKER_CONTACTS = load_contacts(CONTACTS_CSV)

def reload_contacts():
    """Reload globals from disk. Safe to call while contacts_lock is held (RLock)."""
    global ADMIN_NUMBER, WORKER_CONTACTS
    with contacts_lock:                      # RLock: reentrant — no deadlock
        ADMIN_NUMBER, WORKER_CONTACTS = load_contacts(CONTACTS_CSV)

def get_worker_number(name: str):
    return WORKER_CONTACTS.get(name.lower())

# ── Global state ──────────────────────────────────────────
active_camera  = None
active_thread  = None
running        = False
latest_frame   = None
frame_lock     = threading.Lock()
stats = {"total_workers": 0, "compliant": 0, "non_compliant": 0, "last_update": None}
stats_lock = threading.Lock()
violation_log = []
log_lock      = threading.Lock()
alerted_users      = set()
alerted_users_lock = threading.Lock()
face_name_cache      = {}
face_name_cache_lock = threading.Lock()

# =====================================
#  DATABASE
# =====================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        url  TEXT NOT NULL UNIQUE,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id INTEGER, person_name TEXT, violation_type TEXT,
        confidence REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(camera_id) REFERENCES cameras(id)
    )''')
    conn.commit(); conn.close()

init_db()

def get_all_cameras():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT id, name, url, description FROM cameras ORDER BY name")
    rows = c.fetchall(); conn.close()
    return [{"id": r[0], "name": r[1], "url": r[2], "description": r[3]} for r in rows]

# =====================================
#  SMS HELPERS
# =====================================
def _post_sms(number: str, message: str):
    headers = {"authorization": SMS_AUTH, "Content-Type": "application/json"}
    payload = {"route": "q", "message": message, "numbers": number}
    try:
        resp = requests.post(SMS_URL, headers=headers, data=json.dumps(payload), timeout=10)
        print(f"[SMS] -> {number} | {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[SMS] FAILED -> {number}: {e}")

def send_sms_alert(person_name: str, violation_types: list, timestamp: str):
    violations_str = ", ".join(violation_types)
    if ADMIN_NUMBER:
        _post_sms(ADMIN_NUMBER,
                  f"PPE ALERT | Worker: {person_name} | Violations: {violations_str} | Time: {timestamp}")
    worker_number = get_worker_number(person_name)
    if worker_number:
        _post_sms(worker_number,
                  f"Safety Alert: {person_name}, you were detected without "
                  f"{violations_str} at {timestamp}. Please comply with PPE rules immediately.")

def maybe_send_alert(person_name: str, violation_types: list):
    if not person_name or person_name in ("Unknown", "Detecting...", "Error"):
        return
    with alerted_users_lock:
        if person_name in alerted_users:
            return
        alerted_users.add(person_name)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threading.Thread(target=send_sms_alert, args=(person_name, violation_types, ts), daemon=True).start()

# =====================================
#  FACE RECOGNITION THREAD
# =====================================
class FaceRecognitionWorker:
    def __init__(self, raw_frame_ref, raw_frame_lock_ref):
        self._raw_frame = raw_frame_ref
        self._raw_frame_lock = raw_frame_lock_ref
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self): self._thread.start()

    def _loop(self):
        while running:
            time.sleep(0.4)
            with self._raw_frame_lock:
                frame = self._raw_frame[0].copy() if self._raw_frame[0] is not None else None
            if frame is None:
                continue
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            new_cache = {}
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                _, img_enc = cv2.imencode('.jpg', face_crop)
                key = f"{x+w//2}_{y+h//2}"
                try:
                    resp   = requests.post(VERIFY_URL,
                                           files={"file": ("face.jpg", img_enc.tobytes(), "image/jpeg")},
                                           timeout=3)
                    result = resp.json()
                    name   = result.get("name", "Unknown") if result.get("match") else "Unknown"
                except Exception:
                    name = "Error"
                new_cache[key] = name
            with face_name_cache_lock:
                face_name_cache.clear()
                face_name_cache.update(new_cache)

# =====================================
#  BBOX SMOOTHER
# =====================================
class BBoxSmoother:
    def __init__(self, alpha=0.25):
        self.alpha = alpha; self.current = []; self.target = []; self.lock = threading.Lock()

    def update_target(self, detections):
        with self.lock:
            self.target = [(lbl, conf, np.array(box, dtype=float)) for lbl, conf, box in detections]
            if len(self.current) != len(self.target):
                self.current = [(lbl, conf, box.copy()) for lbl, conf, box in self.target]

    def step(self):
        with self.lock:
            result = []
            for i, (lbl, conf, tgt) in enumerate(self.target):
                if i < len(self.current):
                    cur_box = self.current[i][2]
                    new_box = cur_box + self.alpha * (tgt - cur_box)
                    self.current[i] = (lbl, conf, new_box)
                    result.append((lbl, conf, new_box.astype(int).tolist()))
                else:
                    result.append((lbl, conf, tgt.astype(int).tolist()))
            return result

# =====================================
#  PROCESSING LOOP
# =====================================
def processing_loop():
    global latest_frame, running

    # ── Resolve camera source ────────────────────────────────────────────────
    cam_source = active_camera
    if isinstance(active_camera, str) and active_camera.isdigit():
        cam_source = int(active_camera)
    print(f"[CAMERA] Opening source: {cam_source!r}")

    # ── Open capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"[ERROR] cv2.VideoCapture could not open: {cam_source!r}")
        running = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[CAMERA] Opened OK  W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
          f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    # ── Load model ───────────────────────────────────────────────────────────
    try:
        model = YOLO(MODEL_PATH)
        print(f"[MODEL] Loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        cap.release()
        running = False
        return

    raw_frame_container = [None]
    raw_frame_lock = threading.Lock()
    smoother = BBoxSmoother(alpha=0.3)
    FaceRecognitionWorker(raw_frame_container, raw_frame_lock).start()

    # ── Helper ───────────────────────────────────────────────────────────────
    def get_name_for_box(x1, y1, x2, y2, tolerance=80):
        cx = (x1+x2)//2; cy = (y1+y2)//2
        best_name = "Unknown"; best_dist = tolerance
        with face_name_cache_lock:
            for key, name in face_name_cache.items():
                try: kx, ky = map(int, key.split("_"))
                except ValueError: continue
                dist = ((cx-kx)**2 + (cy-ky)**2)**0.5
                if dist < best_dist:
                    best_dist = dist; best_name = name
        return best_name

    # ── Thread 1: capture ────────────────────────────────────────────────────
    def capture_loop():
        consecutive_failures = 0
        while running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures % 50 == 1:      # log every 50 failures, not every frame
                    print(f"[CAPTURE] cap.read() failed (×{consecutive_failures}) — source: {cam_source!r}")
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            with raw_frame_lock:
                raw_frame_container[0] = frame

    # ── Thread 2: inference ──────────────────────────────────────────────────
    def inference_loop():
        while running:
            t0 = time.time()
            with raw_frame_lock:
                frame = raw_frame_container[0].copy() if raw_frame_container[0] is not None else None
            if frame is None:
                time.sleep(0.05); continue
            try:
                results = model(frame, verbose=False, conf=0.4)
            except Exception as e:
                print(f"[INFERENCE] model() error: {e}")
                time.sleep(0.2); continue

            persons = []; violations = []; detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    if cls_id >= len(PPE_CLASSES): continue
                    label = PPE_CLASSES[cls_id]; conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((label, conf, [x1, y1, x2, y2]))
                    if label == "Person":
                        persons.append((x1, y1, x2, y2))
                    elif label.startswith("NO-"):
                        violations.append({"type": label, "conf": conf, "box": (x1, y1, x2, y2)})
            smoother.update_target(detections)
            now_str = datetime.now().strftime("%H:%M:%S")
            for v in violations:
                vx1, vy1, vx2, vy2 = v["box"]
                name = "Unknown"; best_dist = 200
                for (px1, py1, px2, py2) in persons:
                    dist = (((px1+px2)//2-(vx1+vx2)//2)**2 + ((py1+py2)//2-(vy1+vy2)//2)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist; name = get_name_for_box(px1, py1, px2, py2)
                v["person_name"] = name
                if name not in ("Unknown", "Error"):
                    maybe_send_alert(name, [v["type"].replace("NO-", "Missing ")])
            n_persons = len(persons); n_viol = min(len(violations), n_persons)
            with stats_lock:
                stats.update({"total_workers": n_persons, "compliant": n_persons - n_viol,
                               "non_compliant": n_viol, "last_update": datetime.now().isoformat()})
            with log_lock:
                for v in violations:
                    violation_log.append({
                        "timestamp":   now_str,
                        "type":        v["type"].replace("NO-", ""),
                        "person_name": v.get("person_name", "Unknown"),
                        "confidence":  f"{v['conf']:.2f}",
                        "status":      "Critical" if "Hardhat" in v["type"] else "Warning"
                    })
                del violation_log[:-100]
            time.sleep(max(0.0, 1.0 - (time.time() - t0)))

    # ── Thread 3: render ─────────────────────────────────────────────────────
    def render_loop():
        global latest_frame
        while running:
            t0 = time.time()
            with raw_frame_lock:
                frame = raw_frame_container[0].copy() if raw_frame_container[0] is not None else None
            if frame is None:
                time.sleep(0.05); continue
            for label, conf, (x1, y1, x2, y2) in smoother.step():
                color = (0, 0, 255) if label.startswith("NO-") else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display_label = label
                if label == "Person":
                    n = get_name_for_box(x1, y1, x2, y2)
                    if n and n != "Unknown":
                        display_label = n
                cv2.putText(frame, f"{display_label} {conf:.2f}", (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (8, frame.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            with frame_lock:
                latest_frame = frame
            time.sleep(max(0.0, 0.05 - (time.time() - t0)))

    # ── Start threads BEFORE any blocking wait ───────────────────────────────
    # BUG WAS HERE: cap.release() was called before threads started,
    # killing the VideoCapture object that capture_loop depended on.
    t1 = threading.Thread(target=capture_loop,   daemon=True, name="capture")
    t2 = threading.Thread(target=inference_loop, daemon=True, name="inference")
    t3 = threading.Thread(target=render_loop,    daemon=True, name="render")
    t1.start(); t2.start(); t3.start()
    print("[CAMERA] All worker threads started.")

    # ── Block until stopped ──────────────────────────────────────────────────
    while running:
        time.sleep(0.5)

    # ── Cleanup AFTER threads finish ─────────────────────────────────────────
    t1.join(timeout=2); t2.join(timeout=3); t3.join(timeout=2)
    cap.release()   # ← FIXED: release AFTER threads have exited, not before
    print("[CAMERA] Released. Processing loop exited.")

# =====================================
#  STREAM HELPERS
# =====================================
def _encode_frame(frame, quality=75):
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

def _wait_for_frame(timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with frame_lock:
            if latest_frame is not None: return True
        time.sleep(0.05)
    return False

def _mjpeg_generator():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None: time.sleep(0.05); continue
        data = _encode_frame(frame)
        if data:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        time.sleep(0.05)

# =====================================
#  ROUTES
# =====================================
@app.route('/video_feed')
def video_feed():
    if not running:
        return Response("No camera active", status=503)
    if not _wait_for_frame(5):
        return Response("Camera initialising", status=503)
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/snapshot')
def snapshot():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return Response("No frame yet", status=503)
    data = _encode_frame(frame, quality=60)
    return Response(data, mimetype='image/jpeg', headers={'Cache-Control': 'no-cache'})

@app.route('/api/health')
def health():
    with frame_lock:
        has_frame = latest_frame is not None
    return jsonify({"status": "online", "timestamp": datetime.now().isoformat(),
                    "active_camera": active_camera, "processing": running, "has_frame": has_frame})

@app.route('/api/cameras', methods=['GET'])
def list_cameras():
    return jsonify(get_all_cameras())

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    data = request.json or {}
    name = str(data.get("name", "")).strip()
    url  = str(data.get("url",  "")).strip()
    desc = str(data.get("description", "")).strip()
    if not name or not url:
        return jsonify({"error": "name and url required"}), 400
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    try:
        c.execute("INSERT INTO cameras (name,url,description) VALUES (?,?,?)", (name, url, desc))
        conn.commit()
        return jsonify({"message": "Camera added", "id": c.lastrowid}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "URL already exists"}), 409
    finally:
        conn.close()

@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM cameras WHERE id=?", (camera_id,))
    conn.commit(); conn.close()
    return jsonify({"message": "deleted"})

@app.route('/api/select_camera', methods=['POST'])
def select_camera():
    global active_camera, active_thread, running, latest_frame
    data = request.json or {}
    url  = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    running = False
    if active_thread and active_thread.is_alive():
        active_thread.join(timeout=4.0)
    with frame_lock: latest_frame = None
    with stats_lock: stats.update({"total_workers": 0, "compliant": 0, "non_compliant": 0, "last_update": None})
    with log_lock:   violation_log.clear()
    active_camera = url; running = True
    active_thread = threading.Thread(target=processing_loop, daemon=True)
    active_thread.start()
    return jsonify({"message": f"Started: {url}", "status": "starting"})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    global running, active_camera, latest_frame
    running = False; active_camera = None
    with frame_lock: latest_frame = None
    return jsonify({"message": "stopped"})

@app.route('/api/stats')
def get_stats():
    with stats_lock:
        return jsonify(stats.copy())

@app.route('/api/alerts')
def get_alerts():
    with log_lock:
        return jsonify(list(violation_log[-20:]))

@app.route('/api/alerted_users')
def get_alerted_users():
    with alerted_users_lock:
        return jsonify(list(alerted_users))

@app.route('/api/contacts')
def get_contacts():
    result = {"admin": ADMIN_NUMBER[-4:] if ADMIN_NUMBER else None, "workers": {}}
    for name, num in WORKER_CONTACTS.items():
        result["workers"][name] = "****" + num[-4:]
    return jsonify(result)

# ── Register Worker ───────────────────────────────────────
@app.route('/api/register_worker', methods=['POST'])
def register_worker():
    name   = request.form.get("name",   "").strip()
    number = request.form.get("number", "").strip()
    image  = request.files.get("image")

    if not name or not number:
        return jsonify({"error": "name and number are required"}), 400
    if not number.isdigit() or len(number) < 10:
        return jsonify({"error": "Enter a valid 10-digit number"}), 400
    if not image:
        return jsonify({"error": "Worker face image is required"}), 400

    # Save face image
    safe_name  = name.replace(" ", "_")
    image_path = os.path.join(KNOWN_DIR, f"{safe_name}.png")
    try:
        nparr  = np.frombuffer(image.read(), np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return jsonify({"error": "Could not decode image"}), 400
        cv2.imwrite(image_path, img_cv)
    except Exception as e:
        return jsonify({"error": f"Image save failed: {e}"}), 500

    # FIX: Do CSV read/write inside the lock, then call reload_contacts()
    #      OUTSIDE the lock — reload_contacts also acquires contacts_lock
    #      (which is now an RLock, so re-entry from the same thread is also safe).
    updated = False
    with contacts_lock:
        fieldnames, rows = _read_all_rows(CONTACTS_CSV)
        for row in rows:
            if row.get("role", "").lower() == "worker" and row.get("name", "").lower() == name.lower():
                row["name"] = name; row["number"] = number
                updated = True; break
        if not updated:
            rows.append({"role": "worker", "name": name, "number": number})
        _write_all_rows(CONTACTS_CSV, fieldnames, rows)
    # reload_contacts acquires contacts_lock internally — safe because RLock,
    # and called here AFTER releasing the outer with-block.
    reload_contacts()

    verb = "updated" if updated else "registered"
    print(f"[REGISTER] Worker '{name}' {verb}. Image: {image_path}")
    return jsonify({"message": f"Worker '{name}' {verb}", "image": image_path}), 200 if updated else 201


# ── Update Admin Contact ──────────────────────────────────
@app.route('/api/update_admin', methods=['POST'])
def update_admin():
    data   = request.json or {}
    name   = data.get("name", "Site Admin").strip() or "Site Admin"
    number = str(data.get("number", "")).strip()
    if not number.isdigit() or len(number) < 10:
        return jsonify({"error": "Enter a valid 10-digit number"}), 400

    # FIX: same pattern — write inside lock, reload outside.
    updated = False
    with contacts_lock:
        fieldnames, rows = _read_all_rows(CONTACTS_CSV)
        for row in rows:
            if row.get("role", "").lower() == "admin":
                row["name"] = name; row["number"] = number
                updated = True; break
        if not updated:
            rows.insert(0, {"role": "admin", "name": name, "number": number})
        _write_all_rows(CONTACTS_CSV, fieldnames, rows)
    reload_contacts()   # called OUTSIDE the lock block

    print(f"[ADMIN] Contact updated -> {name}: {number}")
    return jsonify({"message": f"Admin contact updated to {number}"}), 200


@app.route('/')
def index():
    return jsonify({"message": "PPE + Face Recognition Backend",
                    "stream": "/video_feed", "snapshot": "/api/snapshot"})


if __name__ == '__main__':
    print("PPE Backend  ->  http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=False)