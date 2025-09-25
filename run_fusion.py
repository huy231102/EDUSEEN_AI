# run_fusion.py  — FastAPI (CPU) for KP+Hands Fusion (webcam) with lifespan API
import os, time, threading, collections, csv, json, asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import base64  # thêm để giải mã ảnh base64 từ FE
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from model_def import KPHandsFusion  # local file

# ===================== ENV CONFIG =====================
MODEL_CKPT  = os.getenv("MODEL_CKPT", "fusion_best.pt")
LABELS_CSV  = os.getenv("LABELS_CSV", "labels.csv")

CAM_INDEX   = int(os.getenv("CAM_INDEX", "0"))
FRAME_W     = int(os.getenv("FRAME_W", "1280"))
FRAME_H     = int(os.getenv("FRAME_H", "720"))           # min 720p
TARGET_FPS  = float(os.getenv("TARGET_FPS", "30"))
T_WINDOW    = int(os.getenv("T_WINDOW", "64"))
IMG_SIZE    = int(os.getenv("HAND_IMG_SIZE", "96"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "2"))
TOP_K_DEF   = int(os.getenv("TOP_K", "5"))
AUTO_START  = os.getenv("AUTO_START", "0") == "1"
API_ONLY    = os.getenv("API_ONLY", "1") == "1"          # 1: no spam console

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]

# Fixed keypoint vector length (pose+hands)
KP_DIM = 258  # 33*(x,y,z,v)=132 + 21*(x,y,z)=63 + 21*(x,y,z)=63

# ===================== LABELS =====================
def load_labels(csv_path: Path) -> List[str]:
    # Support CSV with BOM, or one-label-per-line
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            peek = f.read(2048)
            f.seek(0)
            if "id" in peek and "label" in peek:
                reader = csv.DictReader(f)
                tmp = [(int(r["id"]), r.get("label_name", r.get("label", str(r.get("id", "0"))))) for r in reader if r]
                if not tmp:
                    raise ValueError("Empty CSV")
                n = max(i for i,_ in tmp) + 1
                arr = [""] * n
                for i, nm in tmp:
                    if 0 <= i < n:
                        arr[i] = nm
                for i in range(n):
                    if not arr[i]:
                        arr[i] = str(i)
                return arr
            else:
                f.seek(0)
                return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return [str(i) for i in range(1000)]

# ===================== MODEL =====================
def build_model_from_state_dict(sd: dict) -> KPHandsFusion:
    kp_in_dim = sd["kp_enc.proj.weight"].shape[1]
    kp_d      = sd["kp_enc.proj.weight"].shape[0]
    num_classes = sd["head.4.weight"].shape[0]
    hands_fdim  = sd["hands_enc.frame_encoder.fc.weight"].shape[0]
    hands_tdim  = sd["hands_enc.temporal.layers.0.self_attn.in_proj_weight"].shape[1]
    kp_layers = len({k.split('.')[3] for k in sd.keys() if k.startswith("kp_enc.encoder.layers.")})
    hands_layers = len({k.split('.')[3] for k in sd.keys() if k.startswith("hands_enc.temporal.layers.")})
    m = KPHandsFusion(
        kp_in=kp_in_dim, num_classes=num_classes,
        kp_d=kp_d, kp_layers=kp_layers,
        hands_fdim=hands_fdim, hands_tdim=hands_tdim, hands_layers=hands_layers,
        nhead=8, dropout=0.1
    )
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m

def load_checkpoint(ckpt_path: Path) -> KPHandsFusion:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return build_model_from_state_dict(sd)

# ===================== MEDIAPIPE =====================
def mp_init():
    from mediapipe import solutions as mp
    mp_pose = mp.pose
    mp_hands = mp.hands
    pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False, smooth_landmarks=True)
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0)
    return mp_pose, mp_hands, pose, hands

def extract_keypoints_258(pose_res, hands_res) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray, int, int]:
    vec = []
    has_pose = 0
    num_h = 0

    # Pose (33 landmarks * 4)
    if pose_res and getattr(pose_res, "pose_landmarks", None):
        pls = pose_res.pose_landmarks.landmark
        for lm in pls:
            vec.extend([lm.x, lm.y, lm.z, lm.visibility])
        has_pose = 1
    else:
        vec.extend([0.0] * (33 * 4))

    # Hands L/R (21 * 3 each)
    L = np.zeros((21, 3), dtype=np.float32)
    R = np.zeros((21, 3), dtype=np.float32)
    has_left = 0; has_right = 0

    if hands_res and getattr(hands_res, "multi_hand_landmarks", None):
        lms_list = hands_res.multi_hand_landmarks
        hands_meta = getattr(hands_res, "multi_handedness", [])
        for i in range(len(lms_list)):
            lm_list = lms_list[i]
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
            label = None
            if i < len(hands_meta) and hands_meta[i].classification:
                label = hands_meta[i].classification[0].label.lower()
            if label and label.startswith("left"):
                L = arr; has_left = 1; num_h += 1
            elif label and label.startswith("right"):
                R = arr; has_right = 1; num_h += 1
            else:
                if has_left == 0:
                    L = arr; has_left = 1; num_h += 1
                else:
                    R = arr; has_right = 1; num_h += 1

    vec.extend(L.reshape(-1).tolist())
    vec.extend(R.reshape(-1).tolist())

    vec_np = np.asarray(vec, dtype=np.float32).reshape(-1)
    # lock to KP_DIM
    if vec_np.size != KP_DIM:
        if vec_np.size < KP_DIM:
            vec_np = np.pad(vec_np, (0, KP_DIM - vec_np.size), mode="constant")
        else:
            vec_np = vec_np[:KP_DIM]

    return vec_np, has_pose, num_h, L, R, has_left, has_right

def crop_from_landmarks(frame_bgr: np.ndarray, lms: np.ndarray, img_size: int, margin: float = 1.6) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    if lms is None or (lms==0).all():
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    xs = lms[:,0] * W; ys = lms[:,1] * H
    if np.all(xs==0) and np.all(ys==0):
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cx, cy = xs.mean(), ys.mean()
    r = max((xs.max()-xs.min())/2.0, (ys.max()-ys.min())/2.0) * margin + 1.0
    x1 = int(max(0, cx - r)); y1 = int(max(0, cy - r))
    x2 = int(min(W, cx + r)); y2 = int(min(H, cy + r))
    if x2 <= x1 or y2 <= y1: return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0: return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

# ===================== SLIDING WINDOW =====================
class SlidingWindow:
    def __init__(self, T: int, img_size: int):
        self.T = T; self.img_size = img_size
        self.kp=collections.deque(maxlen=T); self.m_kp=collections.deque(maxlen=T)
        self.left=collections.deque(maxlen=T); self.right=collections.deque(maxlen=T)
        self.m_left=collections.deque(maxlen=T); self.m_right=collections.deque(maxlen=T)

    def push(self, kp_vec, has_pose, left_crop, right_crop, has_left, has_right):
        kv = np.asarray(kp_vec, dtype=np.float32).reshape(-1)
        if kv.size != KP_DIM:
            if kv.size < KP_DIM:
                kv = np.pad(kv, (0, KP_DIM - kv.size), mode="constant")
            else:
                kv = kv[:KP_DIM]
        self.kp.append(kv)
        self.m_kp.append(float(has_pose))
        self.left.append(left_crop.astype(np.uint8)); self.right.append(right_crop.astype(np.uint8))
        self.m_left.append(float(has_left)); self.m_right.append(float(has_right))

    def as_tensors(self):
        t = len(self.kp)
        if t < self.T:
            pad = self.T - t
            zero_kp = np.zeros((KP_DIM,), dtype=np.float32)
            zero_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            for _ in range(pad):
                self.kp.append(zero_kp); self.m_kp.append(0.0)
                self.left.append(zero_img); self.right.append(zero_img)
                self.m_left.append(0.0); self.m_right.append(0.0)
        kp = np.stack(self.kp, axis=0)
        m_kp = np.array(self.m_kp, dtype=np.float32)
        left = np.stack(self.left, axis=0); right = np.stack(self.right, axis=0)
        m_left = np.array(self.m_left, dtype=np.float32); m_right = np.array(self.m_right, dtype=np.float32)

        kp_t = torch.from_numpy(kp).unsqueeze(0)                # [1,T,258]
        m_kp_t = torch.from_numpy(m_kp).unsqueeze(0)            # [1,T]
        left_t = torch.from_numpy(left).permute(0,3,1,2).unsqueeze(0).float().div(255.0)   # [1,T,3,H,W]
        right_t= torch.from_numpy(right).permute(0,3,1,2).unsqueeze(0).float().div(255.0)  # [1,T,3,H,W]
        m_left_t = torch.from_numpy(m_left).unsqueeze(0)        # [1,T]
        m_right_t= torch.from_numpy(m_right).unsqueeze(0)       # [1,T]
        return kp_t, m_kp_t, left_t, right_t, m_left_t, m_right_t

# ===================== APP STATE & LIFESPAN =====================
labels: List[str] = load_labels(Path(LABELS_CSV))
model: Optional[KPHandsFusion] = None

worker_thread: Optional[threading.Thread] = None
stop_event = threading.Event()

pred_lock = threading.Lock()
latest_pred: Dict[str, Any] = {}   # updated every frame

def set_latest_pred(payload: Dict[str, Any]):
    with pred_lock:
        latest_pred.clear()
        latest_pred.update(payload)

def get_latest_pred() -> Dict[str, Any]:
    with pred_lock:
        return dict(latest_pred) if latest_pred else {}

def webcam_worker():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    ok, _ = cap.read()
    if not ok:
        print("[Webcam] Cannot open camera", CAM_INDEX)
        return
    _mp_pose, _mp_hands, pose, hands = mp_init()
    win = SlidingWindow(T_WINDOW, IMG_SIZE)
    frame_idx = 0
    interval = 1.0 / max(1e-6, TARGET_FPS)
    if not API_ONLY:
        print(f"[Runtime] Started. T={T_WINDOW}, target FPS={TARGET_FPS}, capture={int(cap.get(3))}x{int(cap.get(4))}")

    while not stop_event.is_set():
        t0 = time.time()
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hands_res = hands.process(rgb)

        kp_vec, has_pose, _nh, L, R, has_left, has_right = extract_keypoints_258(pose_res, hands_res)
        left_crop  = crop_from_landmarks(frame, L, IMG_SIZE)
        right_crop = crop_from_landmarks(frame, R, IMG_SIZE)
        win.push(kp_vec, has_pose, left_crop, right_crop, has_left, has_right)

        with torch.no_grad():
            kp_t, m_kp_t, left_t, right_t, m_left_t, m_right_t = win.as_tensors()
            logits = model(kp_t, m_kp_t, left_t, right_t, m_left_t, m_right_t)  # [1,C]
            probs_t = F.softmax(logits, dim=-1)[0]
            k = min(TOP_K_DEF, probs_t.numel())
            top_p, top_i = torch.topk(probs_t, k)
            top_pairs = []
            for j in range(k):
                idx = int(top_i[j])
                name = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
                top_pairs.append({"index": idx, "label": name, "prob": float(top_p[j])})

        payload = {
            "ts": time.time(),
            "frame_idx": frame_idx,
            "topk": top_pairs,
            "window": T_WINDOW,
            "capture": {"w": int(cap.get(3)), "h": int(cap.get(4))},
        }
        set_latest_pred(payload)

        if not API_ONLY and frame_idx % PRINT_EVERY == 0:
            head = " | ".join(f"{p['label']}:{p['prob']:.3f}" for p in top_pairs)
            print(f"[Sign] {top_pairs[0]['label']} (p={top_pairs[0]['prob']:.3f})")
            print(f"[Top{k}] {head}")

        dt = time.time() - t0
        if dt < interval: time.sleep(interval - dt)

    cap.release()
    if not API_ONLY:
        print("[Webcam] Stopped.")

def ensure_started():
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    stop_event.clear()
    worker_thread = threading.Thread(target=webcam_worker, daemon=True)
    worker_thread.start()

def ensure_stopped():
    if worker_thread and worker_thread.is_alive():
        stop_event.set()

def _startup_sync():
    global model
    model = load_checkpoint(Path(MODEL_CKPT))
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if AUTO_START:
        ensure_started()

def _shutdown_sync():
    ensure_stopped()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_sync()
    try:
        yield
    finally:
        _shutdown_sync()

app = FastAPI(title="Sign Fusion API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ===================== ROUTES =====================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": len(labels),
        "auto_start": AUTO_START,
        "running": worker_thread.is_alive() if worker_thread else False,
        "fps_target": TARGET_FPS,
        "window": T_WINDOW,
        "img_size": IMG_SIZE,
    }

@app.post("/control/start")
def api_start():
    ensure_started()
    return {"started": True}

@app.post("/control/stop")
def api_stop():
    ensure_stopped()
    return {"stopped": True}

@app.get("/predict/latest")
def api_latest(k: int = TOP_K_DEF):
    snap = get_latest_pred()
    if not snap:
        return JSONResponse({"error": "no_prediction_yet"}, status_code=404)
    if k > 0 and "topk" in snap:
        snap["topk"] = snap["topk"][:min(k, len(snap["topk"]))]
    return snap

@app.get("/predict/stream")
async def api_stream(k: int = TOP_K_DEF, interval_ms: int = 200):
    async def gen():
        while True:
            snap = get_latest_pred()
            if snap:
                if k > 0 and "topk" in snap:
                    snap["topk"] = snap["topk"][:min(k, len(snap["topk"]))]
                yield f"data: {json.dumps(snap)}\n\n"
            await asyncio.sleep(max(0.05, interval_ms/1000))
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.websocket("/ws")
async def ws_feed(ws: WebSocket):
    await ws.accept()
    try:
        try:
            k = int(ws.query_params.get("k", TOP_K_DEF))  # type: ignore
        except Exception:
            k = TOP_K_DEF
        while True:
            snap = get_latest_pred()
            if snap:
                if k > 0 and "topk" in snap:
                    snap["topk"] = snap["topk"][:min(k, len(snap["topk"]))]
                await ws.send_json(snap)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return

# ===================== NEW: Client-stream WebSocket =====================
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    """Nhận ảnh JPEG base64 từ FE, trả về nhãn top-1 dạng "label (prob)"."""
    await ws.accept()
    # Khởi tạo Mediapipe & sliding window riêng cho từng client
    _mp_pose, _mp_hands, pose, hands = mp_init()
    win = SlidingWindow(T_WINDOW, IMG_SIZE)
    try:
        while True:
            data = await ws.receive_text()
            if not data:
                continue
            if data.startswith("data:image"):
                try:
                    b64 = data.split(",", 1)[1] if "," in data else data
                    img_bytes = base64.b64decode(b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_res = pose.process(rgb)
                hands_res = hands.process(rgb)

                kp_vec, has_pose, _nh, L, R, has_left, has_right = extract_keypoints_258(pose_res, hands_res)
                left_crop = crop_from_landmarks(frame, L, IMG_SIZE)
                right_crop = crop_from_landmarks(frame, R, IMG_SIZE)
                win.push(kp_vec, has_pose, left_crop, right_crop, has_left, has_right)

                with torch.no_grad():
                    kp_t, m_kp_t, left_t, right_t, m_left_t, m_right_t = win.as_tensors()
                    logits = model(kp_t, m_kp_t, left_t, right_t, m_left_t, m_right_t)
                    prob_t = F.softmax(logits, dim=-1)[0]
                    top_p, top_i = torch.topk(prob_t, 1)
                    idx = int(top_i[0])
                    label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
                    conf = float(top_p[0])
                await ws.send_text(f"{label} ({conf:.2f})")
    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
