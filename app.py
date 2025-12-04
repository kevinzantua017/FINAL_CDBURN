#!/usr/bin/env python3

import os
import time
import threading
import math
import sqlite3
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

# For admin email & Supabase REST
import smtplib
from email.mime.text import MIMEText
import requests

# --------------------------------------------------------------------------
# Optional LED matrix (luma)
# --------------------------------------------------------------------------
try:
    from luma.core.interface.serial import spi, noop
    from luma.led_matrix.device import max7219
    from luma.core.render import canvas
    from PIL import ImageFont, Image, ImageDraw

    _HAS_LED = True
except Exception:
    _HAS_LED = False

# --------------------------------------------------------------------------
# Flask + SocketIO app
# --------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# --------------------------------------------------------------------------
# Environment / configuration
# --------------------------------------------------------------------------

# Camera IPs (pedestrian, vehicle, traffic light)
CAM_IPS = os.getenv(
    "SC_CAM_IPS",
    "192.168.137.180,192.168.137.23,192.168.137.223",
)
CAM_IPS = [ip.strip() for ip in CAM_IPS.split(",") if ip.strip()]

# Credentials for cameras
CAM_USER = os.getenv("SC_CAM_USER", "Zantua017")
CAM_PASS = os.getenv("SC_CAM_PASS", "Zantua017")

# YOLO model parameters
YOLO_MODEL_MAIN = os.getenv(
    "SC_YOLO_MODEL_MAIN",
    os.getenv("SC_YOLO_MODEL", "yolov8n.pt"),  # backwards compatible
)
YOLO_MODEL_EMERG = os.getenv("SC_YOLO_MODEL_EMERG", "best.pt")

YOLO_CONF = float(os.getenv("SC_YOLO_CONF", "0.25"))
YOLO_IMG_SZ = int(os.getenv("SC_YOLO_IMG", "256"))

# Frame settings
FRAME_W = int(os.getenv("SC_FRAME_W", "256"))
FRAME_H = int(os.getenv("SC_FRAME_H", "144"))
FPS = int(os.getenv("SC_FPS", "10"))
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "8"))
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "60"))
LANE_Y = int(os.getenv("SC_LANE_Y", "120"))

# DB & logging
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))

# Traffic light ROI
TL_ROI_RAW = os.getenv("SC_TL_ROI", "0.35,0.10,0.30,0.60")
try:
    TL_ROI = tuple(map(float, TL_ROI_RAW.split(",")))
except Exception:
    TL_ROI = (0.35, 0.10, 0.60, 0.60)

# Marshal ROI (for color paddle detection on ped cam)
MARSHAL_ROI_RAW = os.getenv("SC_MARSHAL_ROI", "0.10,0.50,0.80,0.45")
try:
    MARSHAL_ROI = tuple(map(float, MARSHAL_ROI_RAW.split(",")))
except Exception:
    MARSHAL_ROI = (0.10, 0.50, 0.80, 0.45)

# Vehicle distance estimate
PPM = float(os.getenv("SC_VEH_PPM", "40.0"))
CLOSE_THRESH_M = float(os.getenv("SC_VEH_CLOSE_M", "6.0"))

# --------------------------------------------------------------------------
# Supabase + admin Gmail configuration (ADMIN APPROVAL)
# --------------------------------------------------------------------------

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://thbxmdojcskayqgfejzn.supabase.co",
)

SUPABASE_SERVICE_KEY = os.getenv(
    "SUPABASE_SERVICE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRoYnhtZG9qY3NrYXlxZ2ZlanpuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjgzMjQ5MywiZXhwIjoyMDc4NDA4NDkzfQ.VjAKaYASTpZXhZy3kmIYQo3WhHZW2nk0Rj63FF_uMyc",
)

# Gmail used to SEND and RECEIVE admin notifications
ADMIN_EMAIL_TO = os.getenv("ADMIN_EMAIL_TO", "Zantua0714@gmail.com")
ADMIN_GMAIL = os.getenv("ADMIN_GMAIL", "Zantua0714@gmail.com")

# Gmail app password (strip spaces in case you typed with spaces)
ADMIN_APP_PASSWORD_RAW = os.getenv("ADMIN_APP_PASSWORD", "ifqf qbxv mvkb vsdk")
ADMIN_APP_PASSWORD = ADMIN_APP_PASSWORD_RAW.replace(" ", "")

# --------------------------------------------------------------------------
# Global state and helpers
# --------------------------------------------------------------------------

latest_frames: Dict[str, Optional[np.ndarray]] = {"ped": None, "veh": None, "tl": None}
latest_jpegs: Dict[str, Optional[bytes]] = {"ped": None, "veh": None, "tl": None}

_state_lock = threading.Lock()
latest_status = {
    "ts": 0.0,
    "ped_count": 0,
    "veh_count": 0,
    "tl_color": "unknown",
    "nearest_vehicle_distance_m": 0.0,
    "avg_vehicle_speed_mps": 0.0,
    "action": "OFF",
    "scenario": "baseline",
    "marshal_signal": "none",
    "board_veh": "OFF",
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
}

board_state = {
    "board_veh": "OFF",
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
    "scenario": "baseline",
    "scenario_1_active": False,
    "scenario_2_active": False,
    "scenario_3_active": False,
    "scenario_4_active": False,
}

# --------------------------------------------------------------------------
# SQLite DB for events
# --------------------------------------------------------------------------


def init_db(path: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ped_count INTEGER,
            veh_count INTEGER,
            tl_color TEXT,
            nearest_vehicle_distance_m REAL,
            avg_vehicle_speed_mps REAL,
            action TEXT,
            scenario TEXT
        );
        """
    )
    con.commit()
    con.close()


def log_event(
    path: str,
    ts: float,
    ped_count: int,
    veh_count: int,
    tl_color: str,
    nearest_m: float,
    avg_mps: float,
    action: str,
    scenario: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts, ped_count, veh_count, tl_color, "
        "nearest_vehicle_distance_m, avg_vehicle_speed_mps, action, scenario) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action, scenario),
    )
    con.commit()
    con.close()


def _db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=3.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn


def _safe_query_rows(sql, args=()):
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(sql, args)
        rows = cur.fetchall()
        conn.close()
        return rows
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return []
        raise


def _candidate_urls_for_ip(ip: str) -> List[str]:
    user = CAM_USER
    pw = CAM_PASS
    rtsp_opts = "?rtsp_transport=udp&fflags=nobuffer&max_delay=0&stimeout=500000"
    templates = [
        f"rtsp://{user}:{pw}@{ip}:554/stream1{rtsp_opts}",
        f"rtsp://{user}:{pw}@{ip}:554/h264{rtsp_opts}",
        f"rtsp://{user}:{pw}@{ip}:554/rtsp.sdp{rtsp_opts}",
        f"http://{user}:{pw}@{ip}/video",
        f"http://{ip}/video",
        f"http://{ip}:8080/video",
    ]
    return templates


class CameraStream:
    def __init__(self, ip: str, width: int, height: int, fps: int, name: str):
        self.ip = ip
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name

        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.stopped = False
        self.cap = None
        self.read_interval = max(0.001, 1.0 / max(1, fps))

        self._open_and_start()

    def _open_capture(self) -> bool:
        urls = _candidate_urls_for_ip(self.ip)
        for url in urls:
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))
                time.sleep(0.3)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    print(f"[CAM-{self.name}] Opened: {url}")
                    return True
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[CAM-{self.name}] Error opening {url}: {e}")

        try:
            cap = cv2.VideoCapture(self.ip)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            if cap is not None and cap.isOpened():
                self.cap = cap
                print(f"[CAM-{self.name}] Fallback open: {self.ip}")
                return True
        except Exception as e:
            print(f"[CAM-{self.name}] Fallback error: {e}")

        self.cap = None
        print(f"[CAM-{self.name}] Failed to open any url for {self.ip}")
        return False

    def _open_and_start(self):
        self._open_capture()
        threading.Thread(target=self._update, daemon=True).start()

    def _reopen(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        time.sleep(0.5)
        self._open_capture()

    def _update(self):
        fail_count = 0
        while not self.stopped:
            ok, frame = False, None
            try:
                if self.cap:
                    for _ in range(2):
                        self.cap.grab()
                    ok, frame = self.cap.retrieve()
                else:
                    ok, frame = False, None
            except Exception:
                ok, frame = False, None

            if ok and frame is not None:
                fail_count = 0
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                fail_count += 1
                if fail_count >= max(10, self.fps):
                    print(f"[CAM-{self.name}] No frames, reconnecting...")
                    self._reopen()
                    fail_count = 0

            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


# --------------------------------------------------------------------------
# YOLO loading
# --------------------------------------------------------------------------
print("[APP] Loading YOLO models (CPU)...")
try:
    from ultralytics import YOLO

    model_main = YOLO(YOLO_MODEL_MAIN)
    try:
        model_main.fuse()
    except Exception:
        pass
    print(f"[APP] Main YOLO loaded: {YOLO_MODEL_MAIN}")
except Exception as e:
    print("[APP] Failed to load main YOLO model:", e)
    model_main = None

try:
    from ultralytics import YOLO as _YOLO2

    model_emerg = _YOLO2(YOLO_MODEL_EMERG)
    try:
        model_emerg.fuse()
    except Exception:
        pass
    print(f"[APP] Emergency/marshal YOLO loaded: {YOLO_MODEL_EMERG}")
except Exception as e:
    print("[APP] Failed to load emergency YOLO model:", e)
    model_emerg = None

VEH_LABELS = {"bicycle", "car", "motorbike", "bus", "truck", "ambulance"}
PED_LABELS = {"person"}
EMERG_VEH_LABELS = {"ambulance", "firetruck", "police"}
MARSHAL_LABEL = "marshal"
PADDLE_LABEL = "paddle"


def infer_main(fr: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
    if model_main is None:
        return []
    r = model_main.predict(fr, conf=YOLO_CONF, imgsz=YOLO_IMG_SZ, verbose=False, max_det=80)
    out = []
    r0 = r[0]
    names = r0.names
    for b in r0.boxes:
        cls_id = int(b.cls[0])
        label = names[cls_id].lower()
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        out.append((label, (x1, y1, x2, y2), conf))
    return out


def infer_emerg(fr: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
    if model_emerg is None:
        return []
    r = model_emerg.predict(fr, conf=YOLO_CONF, imgsz=YOLO_IMG_SZ, verbose=False, max_det=80)
    out = []
    r0 = r[0]
    names = r0.names
    for b in r0.boxes:
        cls_id = int(b.cls[0])
        label = names[cls_id].lower()
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        out.append((label, (x1, y1, x2, y2), conf))
    return out


# --------------------------------------------------------------------------
# Drawing / overlays
# --------------------------------------------------------------------------
def encode_jpg(img: Optional[np.ndarray], q: int = JPEG_QUALITY) -> Optional[bytes]:
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return buf.tobytes() if ok else None


def detect_tl_color(frame: Optional[np.ndarray]) -> str:
    if frame is None:
        return "unknown"
    h, w = frame.shape[:2]
    x = int(w * TL_ROI[0])
    y = int(h * TL_ROI[1])
    ww = int(w * TL_ROI[2])
    hh = int(h * TL_ROI[3])
    roi = frame[y : y + hh, x : x + ww]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def mask(lo, hi):
        return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    red = cv2.bitwise_or(
        mask((0, 120, 120), (10, 255, 255)), mask((170, 120, 120), (180, 255, 255))
    )
    yellow = mask((15, 120, 120), (35, 255, 255))
    green = mask((40, 70, 70), (90, 255, 255))
    counts = {
        "red": int((red > 0).sum()),
        "yellow": int((yellow > 0).sum()),
        "green": int((green > 0).sum()),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 50 else "unknown"


COLOR_RED = (0, 0, 255)
COLOR_GRN = (0, 255, 0)
COLOR_YEL = (0, 255, 255)
COLOR_WHT = (255, 255, 255)
COLOR_MAG = (255, 0, 255)
COLOR_CYAN = (255, 255, 0)


def draw_ped_overlay(img, det_p):
    vis = img.copy()
    cv2.putText(
        vis,
        f"Pedestrians: {len(det_p)}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_GRN,
        1,
    )
    for lab, (x1, y1, x2, y2), conf in det_p:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_GRN, 2)
        cv2.putText(
            vis,
            f"{lab} {conf:.2f}",
            (x1 + 2, max(10, y1 + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_GRN,
            1,
        )
    return vis


def draw_veh_overlay(img, det_v, speeds_by_id=None, det_to_tid=None):
    vis = img.copy()
    cv2.putText(
        vis,
        f"Vehicles: {len(det_v)}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_RED,
        1,
    )
    cv2.line(vis, (0, LANE_Y), (vis.shape[1], LANE_Y), COLOR_YEL, 1)
    for j, (lab, (x1, y1, x2, y2), conf) in enumerate(det_v):
        tid = det_to_tid.get(j) if det_to_tid else None
        speed = speeds_by_id.get(tid, 0.0) if speeds_by_id and tid is not None else 0.0
        kph = speed * 3.6
        if y2 < LANE_Y:
            dist_m = (LANE_Y - y2) / max(1e-6, PPM)
            dist_txt = f"{dist_m:.1f} m"
        else:
            dist_txt = "--"
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RED, 2)
        cv2.putText(
            vis,
            f"{kph:.0f}kph | {dist_txt}",
            (x1 + 2, max(10, y1 + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_RED,
            1,
        )
    return vis


def draw_tl_overlay(img, tl_color):
    vis = img.copy()
    h, w = vis.shape[:2]
    x = int(w * TL_ROI[0])
    y = int(h * TL_ROI[1])
    ww = int(w * TL_ROI[2])
    hh = int(h * TL_ROI[3])
    col = COLOR_WHT
    if tl_color == "red":
        col = COLOR_RED
    elif tl_color == "yellow":
        col = COLOR_YEL
    elif tl_color == "green":
        col = COLOR_GRN
    cv2.rectangle(vis, (x, y), (x + ww, y + hh), col, 2)
    cv2.putText(
        vis,
        tl_color.upper(),
        (x + 4, y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        col,
        1,
    )
    return vis


# --------------------------------------------------------------------------
# Color-based paddle detection (ped cam)
# --------------------------------------------------------------------------
def detect_colored_paddles_ped(frame: Optional[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    if frame is None:
        return []

    h, w = frame.shape[:2]
    rx = int(w * MARSHAL_ROI[0])
    ry = int(h * MARSHAL_ROI[1])
    rw = int(w * MARSHAL_ROI[2])
    rh = int(h * MARSHAL_ROI[3])

    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(w - rx, rw)
    rh = min(h - ry, rh)
    if rw <= 0 or rh <= 0:
        return []

    roi = frame[ry : ry + rh, rx : rx + rw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 160, 160], np.uint8)
    upper_red1 = np.array([8, 255, 255], np.uint8)
    lower_red2 = np.array([172, 160, 160], np.uint8)
    upper_red2 = np.array([179, 255, 255], np.uint8)
    lower_green = np.array([50, 200, 170], np.uint8)
    upper_green = np.array([85, 255, 255], np.uint8)

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_r1, mask_r2)
    mask = cv2.bitwise_or(mask, mask_g)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    roi_area = rw * rh
    min_area = int(roi_area * 0.03)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        aspect = ww / float(hh)
        if aspect < 0.6 or aspect > 1.4:
            continue
        boxes.append((rx + x, ry + y, rx + x + ww, ry + y + hh))

    return boxes


def boxes_overlap(b1: Tuple[int, int, int, int],
                  b2: Tuple[int, int, int, int],
                  min_iou: float = 0.05) -> bool:
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (X2 - X1) * (Y2 - Y1)
    iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
    return iou >= min_iou


# --------------------------------------------------------------------------
# Simple centroid tracker
# --------------------------------------------------------------------------
@dataclass
class Track:
    id: int
    cls: str
    history: deque
    hits: int = 0


class CentroidTracker:
    def __init__(self, max_dist_px=120.0, max_age_s=1.0):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_dist = max_dist_px
        self.max_age = max_age_s

    def update(self, detections, now):
        det_centroids = [
            (lab, ((x1 + x2) // 2, (y1 + y2) // 2)) for lab, (x1, y1, x2, y2) in detections
        ]
        unmatched = list(range(len(det_centroids)))
        det_to_tid = {}

        for tid, tr in list(self.tracks.items()):
            best_j = None
            best_d = 1e9
            t_last, lx, ly = tr.history[-1]
            for j in unmatched:
                lab, (cx, cy) = det_centroids[j]
                if lab != tr.cls:
                    continue
                d = math.hypot(cx - lx, cy - ly)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= self.max_dist:
                lab, (cx, cy) = det_centroids[best_j]
                tr.history.append((now, cx, cy))
                tr.hits += 1
                if len(tr.history) > 30:
                    tr.history.popleft()
                unmatched.remove(best_j)
                det_to_tid[best_j] = tid

        for j in unmatched:
            lab, (cx, cy) = det_centroids[j]
            tr = Track(id=self.next_id, cls=lab, history=deque(maxlen=30))
            tr.history.append((now, cx, cy))
            tr.hits = 1
            self.tracks[tr.id] = tr
            det_to_tid[j] = tr.id
            self.next_id += 1

        to_del = []
        for tid, tr in self.tracks.items():
            t0, _, _ = tr.history[-1]
            if now - t0 > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            self.tracks.pop(tid, None)
        return det_to_tid

    def speeds_mps(self, ppm):
        out = {}
        for tid, tr in self.tracks.items():
            if tr.hits < 3 or len(tr.history) < 2:
                continue
            pts = list(tr.history)[-5:]
            ds = 0.0
            dt = 0.0
            for (t1, x1, y1), (t2, x2, y2) in zip(pts, pts[1:]):
                ds += math.hypot(x2 - x1, y2 - y1)
                dt += (t2 - t1)
            if dt <= 0:
                continue
            out[tid] = (ds / ppm) / dt
        return out


tracker = CentroidTracker()

# --------------------------------------------------------------------------
# LED display helper (SCROLLING TEXT, FULL WORDS)
# --------------------------------------------------------------------------
def _init_led():

    if not _HAS_LED:
        print("[LED] No LED driver; console fallback")

        def _show_led_console(ped_msg: str, veh_msg: str) -> None:
            print(f"[LED] PED: {ped_msg!r} | VEH: {veh_msg!r}")

        return _show_led_console

    # ----------- create MAX7219 devices -----------
    ped_device = None
    veh_device = None

    try:
        serial_ped = spi(port=0, device=0, gpio=noop())
        ped_device = max7219(
            serial_ped,
            cascaded=int(os.getenv("SC_LED_CASCADE", "16")),  # 16 x 8x8 = 128x8
            block_orientation=int(os.getenv("SC_LED_ORIENTATION", "-90")),
            rotate=int(os.getenv("SC_LED_ROTATE", "0")),
        )
        ped_device.contrast(int(os.getenv("SC_LED_BRIGHTNESS", "4")))  # 0–255
        print("[LED] Pedestrian MAX7219 on CE0")
    except Exception as e:
        print("[LED] Failed to init pedestrian board:", repr(e))

    try:
        serial_veh = spi(port=0, device=1, gpio=noop())
        veh_device = max7219(
            serial_veh,
            cascaded=int(os.getenv("SC_LED_CASCADE", "16")),
            block_orientation=int(os.getenv("SC_LED_ORIENTATION", "-90")),
            rotate=int(os.getenv("SC_LED_ROTATE", "0")),
        )
        veh_device.contrast(int(os.getenv("SC_LED_BRIGHTNESS", "4")))
        print("[LED] Vehicle MAX7219 on CE1")
    except Exception as e:
        print("[LED] Failed to init vehicle board:", repr(e))

    if not ped_device and not veh_device:
        print("[LED] No LED devices; console fallback")

        def _show_led_console(ped_msg: str, veh_msg: str) -> None:
            print(f"[LED] PED: {ped_msg!r} | VEH: {veh_msg!r}")

        return _show_led_console

    # ----------- fonts: BIG normal, SMALL for EMERGENCY -----------
    try:
        font_big = None
        font_small = None

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
        ]
        for fp in candidates:
            try:
                # normal text: a bit larger
                font_big = ImageFont.truetype(fp, 12)
                # emergency text: slightly smaller so it fits nicer
                font_small = ImageFont.truetype(fp, 9)
                print(f"[LED] Using font: {fp}")
                break
            except Exception:
                continue

        if font_big is None:
            print("[LED] Falling back to default bitmap font")
            font_big = ImageFont.load_default()
            font_small = font_big
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big

    # ----------- state & geometry -----------
    ped_state = {"msg": ""}
    veh_state = {"msg": ""}

    LOGICAL_W = 64   # logical board width
    LOGICAL_H = 16   # logical board height
    CHAIN_W   = 128  # physical chain width

    def _render_board(device, state, msg: str):
        """Render a steady (non-scrolling) message on one 64x16 board."""
        if device is None:
            return

        msg = (msg or "").upper()

        # Only redraw if text changed (avoids flicker)
        if msg == state["msg"]:
            return
        state["msg"] = msg

        # Choose smaller font when message contains EMERGENCY
        if "EMERGENCY" in msg:
            font = font_small
        else:
            font = font_big

        # Measure text size
        if msg:
            bbox = font.getbbox(msg)  # (left, top, right, bottom)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1] + 4
        else:
            text_w = 0
            text_h = 0

        # 1) Create a 64x16 image and draw centered text
        img = Image.new("1", (LOGICAL_W, LOGICAL_H), 0)
        draw = ImageDraw.Draw(img)

        text_x = max(0, (LOGICAL_W - text_w) // 2)
        # slight upward shift so tall letters don't sit on the very bottom
        text_y = max(-1, (LOGICAL_H - text_h) // 2 - 1)

        if msg:
            draw.text((text_x, text_y), msg, fill=1, font=font)

        # 2) Split into top (0–8) and bottom (8–16)
        top = img.crop((0, 0, LOGICAL_W, 8))
        bottom = img.crop((0, 8, LOGICAL_W, 16))

        # 3) Map to 128x8 chain: [top | bottom]
        frame = Image.new("1", (CHAIN_W, 8), 0)
        frame.paste(top, (0, 0))              # modules 0–7
        frame.paste(bottom, (LOGICAL_W, 0))   # modules 8–15

        device.display(frame)

    def show_led(ped_msg: str = "", veh_msg: str = "") -> None:
        _render_board(ped_device, ped_state, ped_msg)
        _render_board(veh_device, veh_state, veh_msg)

    return show_led



# create global LED updater
show_led = _init_led()








def _encode_and_store(key, frame):
    with _state_lock:
        latest_frames[key] = frame.copy()
        latest_jpegs[key] = encode_jpg(frame, q=JPEG_QUALITY)
    try:
        socketio.emit(f"frame_{key}", {"ts": time.time()}, namespace="/realtime")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Scenario decision
# --------------------------------------------------------------------------
def decide_scenario(
    now_ts,
    ped_count,
    veh_count,
    tl_color,
    nearest_m,
    avg_mps,
    marshal_signal,
    flags,
):
    if board_state.get("scenario_3_active") and flags.get("ambulance", False):
        return ("STOP", "scenario_3_emergency")

    if board_state.get("scenario_4_active") and marshal_signal == "override":
        return ("STOP", "scenario_4_marshall_override")

    if board_state.get("scenario_1_active") and ped_count >= 10 and veh_count <= 10:
        return ("STOP", "scenario_1_night_ped")

    if board_state.get("scenario_2_active") and veh_count >= 10 and ped_count <= 10:
        return ("OFF", "scenario_2_rush_hold")

    if tl_color in ("green", "yellow"):
        return ("STOP", "baseline")
    if tl_color == "red":
        return ("GO", "baseline") if ped_count > 0 else ("OFF", "baseline")
    if ped_count > 0 and (veh_count > 0 or nearest_m < CLOSE_THRESH_M):
        return ("STOP", "baseline")
    return ("OFF", "baseline")


def decide_base_local(ped_count, veh_count, tl_color, nearest_m, avg_mps):
    if tl_color in ("green", "yellow"):
        return "STOP"
    if tl_color == "red":
        return "GO"
    if ped_count > 0 and (veh_count > 0 or nearest_m < CLOSE_THRESH_M):
        return "STOP"
    return "OFF"


def publish_status_from_loop(
    now_ts,
    ped_count,
    veh_count,
    tl_color,
    nearest_m,
    avg_mps,
    action=None,
    scenario=None,
    marshal_signal="none",
    flags=None,
):
    if action is None or scenario is None:
        action, scenario = decide_scenario(
            now_ts,
            ped_count,
            veh_count,
            tl_color,
            nearest_m,
            avg_mps,
            marshal_signal,
            flags or {},
        )
    with _state_lock:
        latest_status.update(
            {
                "ts": now_ts,
                "ped_count": ped_count,
                "veh_count": veh_count,
                "tl_color": tl_color,
                "nearest_vehicle_distance_m": nearest_m,
                "avg_vehicle_speed_mps": avg_mps,
                "action": action,
                "scenario": scenario,
                "marshal_signal": marshal_signal,
            }
        )
        latest_status["board_veh"] = board_state.get("board_veh", "OFF")
        latest_status["board_ped_l"] = board_state.get("board_ped_l", "OFF")
        latest_status["board_ped_r"] = board_state.get("board_ped_r", "OFF")

    # -------- LED messages (FULL WORDS, will scroll) --------
    try:
        ped_msg = "NORMAL"
        veh_msg = "NORMAL"

        if scenario == "scenario_1_night_ped":
            # Pedestrian Priority
            ped_msg = "PED PRIOR"
            veh_msg = "STOP"

        elif scenario == "scenario_2_rush_hold":
            # Vehicle Priority
            ped_msg = "STOP"
            veh_msg = "VEH PRIOR"

        elif scenario == "scenario_3_emergency":
            # Emergency Vehicle Detected – BLINK EMERGENCY
            now = time.time()
            blink_hz = float(os.getenv("SC_LED_EMERG_BLINK_HZ", "2.0"))
            blink_on = int(now * blink_hz) % 2 == 0

            if blink_on:
                ped_msg = "EMERGENCY"
                veh_msg = "EMERGENCY"
            else:
                ped_msg = ""
                veh_msg = ""

        elif scenario == "scenario_4_marshall_override":
            # Marshal Override
            ped_msg = "MARSHALL"
            veh_msg = "MARSHALL"

        show_led(ped_msg, veh_msg)
    except Exception as e:
        print(f"[LED] update error: {e}")

    try:
        socketio.emit("status", latest_status, namespace="/realtime")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Supabase helpers for admin_requests
# --------------------------------------------------------------------------
def supabase_insert_admin_request(email: str, approved: bool) -> bool:
    """
    Insert a row into Supabase admin_requests.
    We *append* rows; check_approved just looks for any approved row.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[ADMIN] Supabase config missing; skip insert")
        return False

    try:
        url = f"{SUPABASE_URL}/rest/v1/admin_requests"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        payload = {"email": email, "approved": bool(approved)}
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        if resp.status_code >= 300:
            print("[ADMIN] Supabase insert failed:", resp.status_code, resp.text)
            return False
        return True
    except Exception as e:
        print("[ADMIN] Supabase insert error:", e)
        return False


def supabase_is_approved(email: str) -> bool:
    """
    Return True if there is at least one row in admin_requests
    with this email and approved = true.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/admin_requests"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        params = {
            "email": f"eq.{email}",
            "approved": "eq.true",
            "select": "approved",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code >= 300:
            print("[ADMIN] Supabase check failed:", resp.status_code, resp.text)
            return False
        data = resp.json()
        if not data:
            return False
        return bool(data[0].get("approved"))
    except Exception as e:
        print("[ADMIN] Supabase check error:", e)
        return False


# --------------------------------------------------------------------------
# Flask routes: streams, status, analytics, logs, scenarios
# --------------------------------------------------------------------------
@app.get("/stream/<cam>")
def stream_cam(cam):
    if cam not in ("ped", "veh", "tl"):
        return "unknown cam", 404

    def mjpeg_stream(key):
        boundary = b"--frame"
        header = b"Content-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
        while True:
            with _state_lock:
                jpg = latest_jpegs.get(key)
            if jpg is None:
                socketio.sleep(0.03)
                continue
            yield boundary + b"\r\n" + header + jpg + b"\r\n"
            socketio.sleep(1.0 / max(1.0, PUBLISH_HZ))

    return Response(
        mjpeg_stream(cam),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status_now")
def api_status_now():
    with _state_lock:
        return jsonify(latest_status)


@app.get("/api/analytics")
def api_analytics():
    now_ts = time.time()
    start_ts = now_ts - 10 * 60
    rows = _safe_query_rows(
        """
        SELECT strftime('%Y-%m-%d %H:%M', datetime(ts, 'unixepoch', 'localtime')) AS minute,
               AVG(ped_count) AS avg_ped,
               AVG(veh_count) AS avg_veh,
               SUM(CASE WHEN action = 'GO' THEN 1 ELSE 0 END) AS go,
               SUM(CASE WHEN action = 'STOP' THEN 1 ELSE 0 END) AS stop,
               SUM(CASE WHEN action = 'OFF' THEN 1 ELSE 0 END) AS off
          FROM events
         WHERE ts >= ?
         GROUP BY minute
         ORDER BY minute ASC
        """,
        (start_ts,),
    )
    data = []
    for minute, avg_ped, avg_veh, go, stop, off in rows:
        data.append(
            {
                "minute": minute,
                "avg_ped": float(avg_ped) if avg_ped is not None else 0.0,
                "avg_veh": float(avg_veh) if avg_veh is not None else 0.0,
                "go": int(go) if go is not None else 0,
                "stop": int(stop) if stop is not None else 0,
                "off": int(off) if off is not None else 0,
            }
        )
    return jsonify(data)


@app.get("/api/logs")
def api_logs():
    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    rows = _safe_query_rows(
        "SELECT id, ts, ped_count, veh_count, tl_color, "
        "nearest_vehicle_distance_m, avg_vehicle_speed_mps, action "
        "FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    result = []
    current_board = {
        "board_veh": latest_status.get("board_veh", "OFF"),
        "board_ped_l": latest_status.get("board_ped_l", "OFF"),
        "board_ped_r": latest_status.get("board_ped_r", "OFF"),
    }
    for row in rows:
        (
            event_id,
            ts_val,
            ped_c,
            veh_c,
            tl_col,
            nearest_m,
            avg_mps,
            action_val,
        ) = row
        result.append(
            {
                "id": event_id,
                "ts": ts_val,
                "ped_count": ped_c,
                "veh_count": veh_c,
                "tl_color": tl_col,
                "nearest_vehicle_distance_m": nearest_m,
                "avg_vehicle_speed_mps": avg_mps,
                "action": action_val,
                "board_veh": current_board["board_veh"],
                "board_ped_l": current_board["board_ped_l"],
                "board_ped_r": current_board["board_ped_r"],
            }
        )
    return jsonify(result)


@app.post("/api/set_scenario")
def api_set_scenario():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}
    valid_keys = {
        "board_veh",
        "board_ped_l",
        "board_ped_r",
        "scenario",
        "scenario_1_active",
        "scenario_2_active",
        "scenario_3_active",
        "scenario_4_active",
    }
    updated = False
    with _state_lock:
        for k, v in data.items():
            if k in valid_keys:
                if k.endswith("_active"):
                    board_state[k] = bool(v)
                else:
                    board_state[k] = (
                        str(v).upper() if k.startswith("board_") else str(v)
                    )
                updated = True
    if updated:
        publish_status_from_loop(
            time.time(),
            latest_status.get("ped_count", 0),
            latest_status.get("veh_count", 0),
            latest_status.get("tl_color", "unknown"),
            latest_status.get("nearest_vehicle_distance_m", 0.0),
            latest_status.get("avg_vehicle_speed_mps", 0.0),
            marshal_signal=latest_status.get("marshal_signal", "none"),
            flags={},
        )
    return jsonify({"ok": True, "board_state": board_state})


@app.post("/api/logs/delete")
def api_logs_delete():
    try:
        data = request.get_json(force=True) or {}
        ids = [int(i) for i in data.get("ids", [])]
    except Exception:
        ids = []
    if not ids:
        return jsonify({"ok": True, "deleted": 0})
    placeholders = ",".join(["?"] * len(ids))
    conn = _db_connect()
    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids)
        conn.commit()
        deleted = cur.rowcount
    except Exception:
        deleted = 0
    finally:
        conn.close()
    return jsonify({"ok": True, "deleted": deleted})


@app.post("/api/logs/clear")
def api_logs_clear():
    try:
        data = request.get_json(force=True) or {}
        do_clear = bool(data.get("all"))
    except Exception:
        do_clear = False
    if not do_clear:
        return jsonify({"ok": False, "deleted": 0})
    conn = _db_connect()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM events")
        conn.commit()
        deleted = cur.rowcount
    except Exception:
        deleted = 0
    finally:
        conn.close()
    return jsonify({"ok": True, "deleted": deleted})


@app.post("/api/set_quality")
def api_set_quality():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}
    qmap = {
        "low": 40,
        "medium": 60,
        "high": 85,
    }
    qual = str(data.get("quality", "medium")).lower()
    new_q = qmap.get(qual, 60)
    global JPEG_QUALITY
    JPEG_QUALITY = new_q
    with _state_lock:
        for key in latest_jpegs.keys():
            latest_jpegs[key] = None
    return jsonify({"ok": True, "quality": qual, "jpeg_quality": new_q})


# --------------------------------------------------------------------------
# NEW: Admin approval endpoints (Option A: OTP + approval)
# --------------------------------------------------------------------------
@app.post("/api/request_login")
def api_request_login():
    """
    Frontend calls this when user enters their email (NOT yet approved).
    - Inserts row {email, approved=false} in Supabase (optional)
    - Sends Gmail to admin with Approve / Deny links
    """
    try:
        data = request.get_json(force=True) or {}
        email = (data.get("email") or "").strip()
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    if not email:
        return jsonify({"ok": False, "error": "Missing email"}), 400

    supabase_insert_admin_request(email, approved=False)

    base = request.url_root.rstrip("/")
    approve_link = f"{base}/api/approve_user?email={email}"
    deny_link = f"{base}/api/deny_user?email={email}"

    body = (
        f"User wants to log in: {email}\n\n"
        f"ACCEPT: {approve_link}\n"
        f"DENY:   {deny_link}\n"
    )

    msg = MIMEText(body)
    msg["Subject"] = "Login Request – LEDBoard"
    msg["From"] = ADMIN_GMAIL
    msg["To"] = ADMIN_EMAIL_TO

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ADMIN_GMAIL, ADMIN_APP_PASSWORD)
            smtp.sendmail(ADMIN_GMAIL, [ADMIN_EMAIL_TO], msg.as_string())
    except Exception as e:
        print("[ADMIN] Failed to send email:", e)
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Failed to send email to admin",
                    "detail": str(e),
                }
            ),
            500,
        )

    return jsonify({"ok": True})


@app.get("/api/check_approved")
def api_check_approved():
    """
    Frontend calls this BEFORE sending OTP.
    If True → frontend calls supabase.auth.signInWithOtp(email)
    If False → frontend calls /api/request_login
    """
    email = (request.args.get("email") or "").strip()
    if not email:
        return jsonify({"approved": False})
    approved = supabase_is_approved(email)
    return jsonify({"approved": bool(approved)})


@app.get("/api/approve_user")
def api_approve_user():
  
    email = (request.args.get("email") or "").strip()
    if not email:
        return "Missing email", 400

    ok = supabase_insert_admin_request(email, approved=True)
    if not ok:
        return "Error updating approval in Supabase", 500

    return f"User APPROVED: {email}"


@app.get("/api/deny_user")
def api_deny_user():
    
    email = (request.args.get("email") or "").strip()
    if not email:
        return "Missing email", 400

    supabase_insert_admin_request(email, approved=False)
    return f"User DENIED: {email}"


# --------------------------------------------------------------------------
# Main inference pipeline
# --------------------------------------------------------------------------
def pipeline_main():
    init_db(DB_PATH)
    roles = ["ped", "veh", "tl"]
    cams = {}
    for idx, role in enumerate(roles):
        ip = CAM_IPS[idx] if idx < len(CAM_IPS) else CAM_IPS[0]
        cs = CameraStream(ip, FRAME_W, FRAME_H, FPS, name=role)
        cams[role] = cs
        print(f"[MAIN] Started camera {role} -> {ip}")

    last_log_ts = 0.0

    while True:
        t0 = time.time()

        rp, fp = cams["ped"].read()
        rv, fv = cams["veh"].read()
        rt, ft = cams["tl"].read()

        blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        fp_s = fp if rp and fp is not None else blank
        fv_s = fv if rv and fv is not None else blank
        ft_s = ft if rt and ft is not None else blank

        ped_count = 0
        veh_count = 0
        nearest_m = float("inf")
        avg_mps = 0.0

        emerg_vehicle_present = False
        marshal_override_trigger = False

        det_p = []
        det_v = []
        det_to_tid = {}

        # Pedestrians
        person_boxes_ped: List[Tuple[int, int, int, int]] = []
        if fp_s is not None:
            try:
                det_full_ped = infer_main(fp_s)
                det_p = [
                    (lab, box, cf) for lab, box, cf in det_full_ped if lab in PED_LABELS
                ]
                ped_count = len(det_p)
                person_boxes_ped = [box for _, box, _ in det_p]
            except Exception as e:
                print("[PIPE] Ped detection error:", e)
                det_p = []
                ped_count = 0
                person_boxes_ped = []

        # Vehicles
        if fv_s is not None:
            try:
                det_full_v = infer_main(fv_s)
                det_v = [
                    (lab, box, cf) for lab, box, cf in det_full_v if lab in VEH_LABELS
                ]
                veh_count = len(det_v)

                now_ts = time.time()
                det_to_tid = tracker.update(
                    [(lab, box) for lab, box, _ in det_v], now_ts
                )
                speed_map = tracker.speeds_mps(PPM)
                if speed_map:
                    avg_mps = float(np.mean(list(speed_map.values())))
                for lab, (x1, y1, x2, y2), _ in det_v:
                    if y2 < LANE_Y:
                        dist = (LANE_Y - y2) / max(1e-6, PPM)
                    else:
                        dist = float("inf")
                    nearest_m = min(nearest_m, dist)
            except Exception as e:
                print("[PIPE] Vehicle detection error:", e)
                det_v = []
                veh_count = 0

        if nearest_m == float("inf"):
            nearest_m = 0.0

        marshal_boxes_ped: List[Tuple[int, int, int, int]] = []
        paddle_boxes_ped_yolo: List[Tuple[int, int, int, int]] = []
        paddle_boxes_ped_color: List[Tuple[int, int, int, int]] = []
        paddle_boxes_ped_all: List[Tuple[int, int, int, int]] = []

        try:
            emerg_ped = infer_emerg(fp_s)
            for lab, box, _ in emerg_ped:
                if lab == MARSHAL_LABEL:
                    marshal_boxes_ped.append(box)
                elif lab == PADDLE_LABEL:
                    paddle_boxes_ped_yolo.append(box)

            paddle_boxes_ped_color = detect_colored_paddles_ped(fp_s)
            paddle_boxes_ped_all = paddle_boxes_ped_yolo + paddle_boxes_ped_color

            emerg_veh = infer_emerg(fv_s)
            if emerg_veh:
                if any(lab in EMERG_VEH_LABELS for lab, _, _ in emerg_veh):
                    emerg_vehicle_present = True

            if marshal_boxes_ped and paddle_boxes_ped_all:
                marshal_override_trigger = True

            if not marshal_override_trigger and paddle_boxes_ped_all and person_boxes_ped:
                for pb in paddle_boxes_ped_all:
                    for per in person_boxes_ped:
                        if boxes_overlap(pb, per):
                            marshal_override_trigger = True
                            break
                    if marshal_override_trigger:
                        break

            if not marshal_override_trigger and marshal_boxes_ped and person_boxes_ped:
                for mb in marshal_boxes_ped:
                    for per in person_boxes_ped:
                        if boxes_overlap(mb, per):
                            marshal_override_trigger = True
                            break
                    if marshal_override_trigger:
                        break

        except Exception as e:
            print("[PIPE] Emergency/marshal detection error:", e)
            emerg_vehicle_present = False
            marshal_override_trigger = False

        tl_color = detect_tl_color(ft_s)
        marshal_signal = "override" if marshal_override_trigger else "none"

        flags = {
            "ambulance": emerg_vehicle_present,
            "night": time.localtime().tm_hour >= 21,
            "rush": time.localtime().tm_hour == 7,
        }

        action, scenario = decide_scenario(
            time.time(),
            ped_count,
            veh_count,
            tl_color,
            nearest_m,
            avg_mps,
            marshal_signal,
            flags,
        )

        vp = draw_ped_overlay(fp_s, det_p)
        vv = draw_veh_overlay(
            fv_s, det_v, speeds_by_id=tracker.speeds_mps(PPM), det_to_tid=det_to_tid
        )
        vt = draw_tl_overlay(ft_s, tl_color)

        try:
            for box in marshal_boxes_ped:
                x1, y1, x2, y2 = box
                cv2.rectangle(vp, (x1, y1), (x2, y2), COLOR_MAG, 1)
                cv2.putText(
                    vp,
                    "MARSHAL",
                    (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_MAG,
                    1,
                )
            for box in paddle_boxes_ped_all:
                x1, y1, x2, y2 = box
                cv2.rectangle(vp, (x1, y1), (x2, y2), COLOR_CYAN, 1)
                cv2.putText(
                    vp,
                    "PADDLE",
                    (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_CYAN,
                    1,
                )
        except Exception as e:
            print("[PIPE] Debug overlay error:", e)

        _encode_and_store("ped", vp)
        _encode_and_store("veh", vv)
        _encode_and_store("tl", vt)

        publish_status_from_loop(
            now_ts=time.time(),
            ped_count=ped_count,
            veh_count=veh_count,
            tl_color=tl_color,
            nearest_m=nearest_m,
            avg_mps=avg_mps,
            action=action,
            scenario=scenario,
            marshal_signal=marshal_signal,
            flags=flags,
        )

        now = time.time()
        if now - last_log_ts >= LOG_EVERY_SEC:
            log_event(
                DB_PATH,
                now,
                ped_count,
                veh_count,
                tl_color,
                nearest_m,
                avg_mps,
                action,
                scenario,
            )
            last_log_ts = now

        elapsed = time.time() - t0
        target = 1.0 / max(1, FPS)
        if elapsed < target:
            time.sleep(target - elapsed)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    init_db(DB_PATH)

    t = threading.Thread(target=pipeline_main, daemon=True)
    t.start()

    print("[APP] Starting SocketIO server on 0.0.0.0:5000 (eventlet)")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
