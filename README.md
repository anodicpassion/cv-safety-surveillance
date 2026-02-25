<div align="center">

# 🦺 CV Safety Surveillance

**Real-time PPE compliance monitoring and identity-aware violation alerting for industrial environments**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAB?style=flat-square)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](LICENSE)

</div>

---

## Overview

CV Safety Surveillance is a **computer-vision-driven safety monitoring system** designed for construction sites, factories, and any industrial environment requiring PPE compliance enforcement. The system fuses a **YOLOv8 object detection pipeline** with **real-time face recognition** to identify individual workers, log PPE violations, and dispatch targeted SMS alerts — all over a live MJPEG video stream served via a REST API.

The architecture decouples capture, inference, face recognition, and rendering into independent threads, enabling near-real-time throughput on commodity hardware without sacrificing frame continuity.

---

## Key Features

- **Multi-class PPE Detection** — Detects 10 object classes: `Hardhat`, `Mask`, `Safety Vest`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `machinery`, `vehicle` using a fine-tuned YOLOv8 model.
- **Identity-Aware Violation Tracking** — Associates detected violations with named individuals via proximity matching between `Person` bounding boxes and a remote face-verification endpoint.
- **Automated SMS Alerting** — On first violation detection per shift, dispatches role-specific SMS to both the offending worker and the site admin via the Fast2SMS Bulk API.
- **Live MJPEG Stream** — Exposes a continuous `multipart/x-mixed-replace` video feed with annotated bounding boxes, identity labels, and timestamps rendered directly onto frames.
- **Multi-Camera Management** — Cameras are registered, selected, and deleted via REST endpoints; metadata is persisted in a local SQLite database.
- **Worker Registration** — Registers workers (name, mobile, face image) through a multipart form API; updates `contacts.csv` and reloads in-memory contact state atomically.
- **Thread-Safe State Management** — All shared state (frames, stats, violation log, alert set, face cache) is guarded by dedicated `threading.Lock` / `threading.RLock` primitives.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Flask REST API                      │
│  /video_feed  /api/stats  /api/alerts  /api/cameras ...  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   processing_loop   │  (per active camera)
              └──┬────────┬────────┘
                 │        │
        ┌────────▼──┐  ┌──▼──────────┐  ┌──────────────┐
        │  capture  │  │  inference  │  │    render    │
        │  Thread   │  │   Thread    │  │   Thread     │
        │           │  │  YOLOv8     │  │  cv2 draw    │
        │ cap.read()│  │  + smoother │  │  + MJPEG out │
        └─────┬─────┘  └──────┬──────┘  └──────────────┘
              │               │
              └──────┬────────┘
                     │ raw_frame_container (shared buffer)
              ┌──────▼──────────────┐
              │ FaceRecognitionWorker│
              │  Haar cascade detect │
              │  → POST /verify      │  (remote ngrok endpoint)
              │  → face_name_cache   │
              └─────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Object Detection** | [Ultralytics YOLOv8](https://ultralytics.com) | Fine-tuned PPE detection model (`ppe.pt`) |
| **Computer Vision** | [OpenCV 4](https://opencv.org) | Frame capture, Haar cascade face detection, JPEG encoding, bounding box rendering |
| **Web Framework** | [Flask 3](https://flask.palletsprojects.com) + [Flask-CORS](https://flask-cors.readthedocs.io) | REST API, MJPEG streaming, multipart form handling |
| **Face Recognition** | Remote inference via [ngrok](https://ngrok.com) tunnel | Stateless HTTP face verification endpoint |
| **SMS Gateway** | [Fast2SMS Bulk API v2](https://fast2sms.com) | Worker and admin violation notifications |
| **Database** | SQLite 3 (via `sqlite3` stdlib) | Camera registry and violation log persistence |
| **Concurrency** | `threading` stdlib — `Lock`, `RLock`, `Thread` | Thread-safe multi-producer/consumer pipeline |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript | Dashboard UI, live feed, stats, worker registration |
| **Serialisation** | CSV (`csv.DictReader/Writer`) + JSON | Contact management, API responses |

---

## Project Structure

```
cv-safety-surveillance/
├── backend/
│   ├── app.py               # Flask application — API, processing pipeline, stream
│   ├── ppe_monitor.db       # SQLite database (auto-created)
│   ├── contacts.csv         # Admin + worker contact registry
│   ├── models/
│   │   └── ppe.pt           # YOLOv8 fine-tuned PPE weights
│   └── known/               # Worker face images (PNG, named by worker)
└── frontend/
    └── index.html           # Single-page dashboard
```

---

## API Reference

### Camera Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras` | List all registered cameras |
| `POST` | `/api/cameras` | Register a new camera `{name, url, description}` |
| `DELETE` | `/api/cameras/<id>` | Remove a camera by ID |
| `POST` | `/api/select_camera` | Activate a camera stream `{url}` |
| `POST` | `/api/stop_camera` | Stop the active stream |

### Monitoring

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/video_feed` | Live MJPEG stream (`multipart/x-mixed-replace`) |
| `GET` | `/api/snapshot` | Latest frame as JPEG |
| `GET` | `/api/stats` | Worker counts — total / compliant / non-compliant |
| `GET` | `/api/alerts` | Last 20 violation events |
| `GET` | `/api/health` | Service health and stream status |

### Personnel

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/register_worker` | Register worker with face image (multipart) |
| `POST` | `/api/update_admin` | Update site admin contact number |
| `GET` | `/api/contacts` | List masked contact registry |
| `GET` | `/api/alerted_users` | Workers who received alerts this session |

---

## Detection Pipeline

```
cap.read()
    │
    ▼
raw_frame_container          ← shared memory buffer (thread-safe)
    │
    ├──► YOLOv8 inference    → [Person, NO-Hardhat, NO-Mask, NO-Safety Vest, ...]
    │         │
    │         ├──► BBoxSmoother.update_target()   ← exponential moving average (α=0.3)
    │         │
    │         └──► violation → Person proximity match (Euclidean centroid distance)
    │                              │
    │                              └──► get_name_for_box() → face_name_cache lookup
    │                                        │
    │                                        └──► maybe_send_alert() → Fast2SMS
    │
    └──► FaceRecognitionWorker (0.4s cadence)
              │
              ├──► Haar cascade detectMultiScale()
              └──► POST face crop → /verify → {match: bool, name: str}
                        │
                        └──► face_name_cache update
```

**Bounding box smoothing** uses an exponential moving average over box coordinates to eliminate jitter between inference frames, producing visually stable overlays without adding latency.

---

## Concurrency Model

The pipeline runs four concurrent threads per active camera, coordinated via shared memory buffers rather than queues to minimise latency:

```
Thread          Lock(s) held              Shared resource
─────────────── ──────────────────────    ──────────────────────────
capture_loop    raw_frame_lock (write)    raw_frame_container[0]
inference_loop  raw_frame_lock (read)     raw_frame_container[0]
                stats_lock (write)        stats dict
                log_lock (write)          violation_log list
render_loop     raw_frame_lock (read)     raw_frame_container[0]
                frame_lock (write)        latest_frame
FaceRecWorker   raw_frame_lock (read)     raw_frame_container[0]
                face_name_cache_lock(w)   face_name_cache dict
```

`contacts_lock` is an `RLock` (reentrant) to allow `reload_contacts()` — which itself acquires the lock — to be called safely from within routes that already hold it.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A YOLOv8-compatible PPE model (`ppe.pt`) placed at `backend/models/ppe.pt`
- A running face-verification endpoint (e.g. a Colab notebook exposed via ngrok returning `{"match": bool, "name": str}`)
- Fast2SMS account and API key

### Installation

```bash
git clone https://github.com/anodicpassion/cv-safety-surveillance.git
cd cv-safety-surveillance/backend

pip install flask flask-cors opencv-python ultralytics requests numpy
```

### Configuration

Edit the constants at the top of `app.py`:

```python
MODEL_PATH = "models/ppe.pt"          # Path to YOLOv8 weights
VERIFY_URL = "https://<ngrok-url>/verify"  # Face verification endpoint
SMS_AUTH   = "<your-fast2sms-api-key>"
```

Populate `contacts.csv`:

```csv
role,name,number
admin,Site Admin,9000000000
worker,Rahul Sharma,9111111111
```

### Run

```bash
python app.py
# Backend available at http://0.0.0.0:5001
```

Open `frontend/index.html` in a browser, or serve it via any static file server pointing to the backend at `http://localhost:5001`.

### Register a Worker via API

```bash
curl -X POST http://localhost:5001/api/register_worker \
  -F "name=Rahul Sharma" \
  -F "number=9111111111" \
  -F "image=@rahul.jpg"
```

---

## PPE Classes

| Class ID | Label | Alert Severity |
|---|---|---|
| 0 | `Hardhat` | — |
| 1 | `Mask` | — |
| 2 | `NO-Hardhat` | 🔴 Critical |
| 3 | `NO-Mask` | 🟡 Warning |
| 4 | `NO-Safety Vest` | 🟡 Warning |
| 5 | `Person` | — |
| 6 | `Safety Cone` | — |
| 7 | `Safety Vest` | — |
| 8 | `machinery` | — |
| 9 | `vehicle` | — |

---

## License

Distributed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for full terms.

---

<div align="center">

Built with OpenCV · YOLOv8 · Flask · Fast2SMS

</div>