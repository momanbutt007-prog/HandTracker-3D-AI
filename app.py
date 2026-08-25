import math
import time
import threading
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="HandTracker 3D AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
MODEL_PATH = str(APP_DIR / "hand_landmarker.task")

SHAPES = [
    "Cube",
    "Pyramid",
    "Sphere",
    "Cylinder",
    "Cone",
    "Torus",
    "Icosahedron",
]

SHAPE_ICONS = {
    "Cube": "🧊",
    "Pyramid": "🔺",
    "Sphere": "🔵",
    "Cylinder": "🥫",
    "Cone": "🍦",
    "Torus": "🍩",
    "Icosahedron": "💎",
}

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9


# ============================================================
# THEME / CSS  (orange + cyan "vision engine" palette, polished)
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

#MainMenu, footer { visibility: hidden; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

/* ---------- App background ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(255, 107, 53, 0.10) 0%, transparent 40%),
        radial-gradient(circle at 92% 8%, rgba(0, 212, 255, 0.10) 0%, transparent 48%),
        radial-gradient(circle at 50% 105%, rgba(90, 210, 120, 0.05) 0%, transparent 55%),
        linear-gradient(165deg, #05080d 0%, #0a0f16 45%, #060a10 100%);
    color: #eef3f8;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 25% 0%, rgba(255,107,53,0.08), transparent 45%),
        radial-gradient(circle at 90% 45%, rgba(0,212,255,0.08), transparent 50%),
        linear-gradient(180deg, #0a0f16 0%, #050810 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * { color: #eef3f8 !important; }

.sb-logo-wrap { text-align: center; padding: 0.6rem 0 0.4rem 0; }

.sb-logo-badge {
    width: 58px;
    height: 58px;
    margin: 0 auto 0.5rem auto;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    background: linear-gradient(135deg, #ff6b35, #ff9142 55%, #00d4ff);
    box-shadow: 0 10px 26px rgba(255,107,53,0.30);
}

.sb-title {
    font-size: 1.18rem;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    margin-bottom: 0.1rem;
}

.sb-subtitle {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.sb-section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    margin: 0.5rem 0 0.6rem 0;
    color: #ffffff !important;
}

.sb-section-label .icon-chip-sm {
    width: 25px;
    height: 25px;
    min-width: 25px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    background: linear-gradient(135deg, rgba(255,107,53,0.28), rgba(0,212,255,0.28));
    border: 1px solid rgba(255,255,255,0.12);
}

.sb-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

.sb-card p { margin: 0.25rem 0; font-size: 0.85rem; color: rgba(255,255,255,0.8) !important; }
.sb-card b { color: #ffb088 !important; }

.sb-shape-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.85rem;
}

.sb-status-chip {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(45,212,120,0.08);
    border: 1px solid rgba(45,212,120,0.28);
    border-radius: 12px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #86efac !important;
}

/* Selectbox in sidebar */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
}

/* ---------- Hero banner ---------- */
.hero {
    padding: 2.1rem 2.4rem;
    border-radius: 22px;
    margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #7c2d12 0%, #b8451f 40%, #0e7490 100%);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 20px 45px rgba(120, 60, 20, 0.30);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}

.hero h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.hero p {
    color: rgba(255,255,255,0.82);
    max-width: 850px;
    font-size: 0.95rem;
    margin-bottom: 0.9rem;
}

.badge {
    display: inline-block;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    background: rgba(0,212,255,0.14);
    border: 1px solid rgba(0,212,255,0.35);
    color: #7be0ff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ---------- Section header row ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #ffffff;
    margin: 0.3rem 0 0.7rem 0;
}

.section-header .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: linear-gradient(120deg, #ff6b35, #00d4ff);
    box-shadow: 0 0 10px rgba(255,107,53,0.7);
}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(165deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1rem 1.1rem 0.7rem 1.1rem;
    box-shadow: 0 10px 26px rgba(0,0,0,0.30);
}

[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.62) !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Poppins', sans-serif; }

/* ---------- Tech stack cards ---------- */
.tech-card {
    background: linear-gradient(165deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.28);
    height: 100%;
}

.tech-card .tech-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
.tech-card b { color: #ffffff; }
.tech-card p { color: rgba(255,255,255,0.65); font-size: 0.85rem; margin: 0.2rem 0 0 0; }

/* ---------- WebRTC video frame ---------- */
[data-testid="stWebRtcStreamer"] {
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.4);
}

/* ---------- Buttons ---------- */
.stButton button {
    border-radius: 10px;
    background: linear-gradient(120deg, #b8451f, #0e7490);
    color: white;
    border: none;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(184,69,31,0.30);
}

/* ---------- Expander / how it works ---------- */
div[data-testid="stExpander"] {
    background: linear-gradient(165deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
}

hr { border-color: rgba(255,255,255,0.08) !important; }

.footer-caption {
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-size: 0.8rem;
    padding-top: 0.6rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🖐️ HandTracker 3D AI</h1>
    <p>
        Real-time computer vision system that transforms hand movements
        into interactive 3D object controls using MediaPipe, OpenCV,
        NumPy and WebRTC.
    </p>
    <div class="badge">● LIVE COMPUTER VISION SYSTEM</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 3D SHAPES
# ============================================================

def cube():
    vertices = np.array([
        [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
    ], dtype=np.float32)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    return vertices, edges


def pyramid():
    vertices = np.array([
        [-1,1,-1], [1,1,-1], [1,1,1],
        [-1,1,1], [0,-1,0]
    ], dtype=np.float32)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (0,4),(1,4),(2,4),(3,4)
    ]

    return vertices, edges


def sphere():
    vertices = []
    edges = []

    latitudes = 10
    longitudes = 16

    for i in range(latitudes + 1):
        theta = math.pi * i / latitudes

        for j in range(longitudes):
            phi = 2 * math.pi * j / longitudes

            vertices.append([
                math.sin(theta) * math.cos(phi),
                math.cos(theta),
                math.sin(theta) * math.sin(phi)
            ])

    for i in range(latitudes + 1):
        for j in range(longitudes):

            current = i * longitudes + j
            next_j = i * longitudes + (j + 1) % longitudes

            edges.append((current, next_j))

            if i < latitudes:
                edges.append(
                    (current, (i + 1) * longitudes + j)
                )

    return np.array(vertices, dtype=np.float32), edges


def cylinder():
    vertices = []
    edges = []
    sides = 16

    for y in [-1, 1]:
        for i in range(sides):
            a = 2 * math.pi * i / sides
            vertices.append([math.cos(a), y, math.sin(a)])

    for i in range(sides):
        j = (i + 1) % sides
        edges += [
            (i, j),
            (sides + i, sides + j),
            (i, sides + i)
        ]

    return np.array(vertices, dtype=np.float32), edges


def cone():
    vertices = []
    edges = []
    sides = 16

    for i in range(sides):
        a = 2 * math.pi * i / sides
        vertices.append([math.cos(a), 1, math.sin(a)])

    vertices.append([0, -1, 0])
    apex = sides

    for i in range(sides):
        edges.append((i, (i + 1) % sides))
        edges.append((i, apex))

    return np.array(vertices, dtype=np.float32), edges


def torus():
    vertices = []
    edges = []

    major = 14
    minor = 8
    R = 0.7
    r = 0.3

    for i in range(major):
        theta = 2 * math.pi * i / major

        for j in range(minor):
            phi = 2 * math.pi * j / minor

            vertices.append([
                (R + r * math.cos(phi)) * math.cos(theta),
                r * math.sin(phi),
                (R + r * math.cos(phi)) * math.sin(theta)
            ])

    for i in range(major):
        for j in range(minor):

            current = i * minor + j

            edges.append((
                current,
                i * minor + (j + 1) % minor
            ))

            edges.append((
                current,
                ((i + 1) % major) * minor + j
            ))

    return np.array(vertices, dtype=np.float32), edges


def icosahedron():

    t = (1 + math.sqrt(5)) / 2

    vertices = np.array([
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
        [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
        [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]
    ], dtype=np.float32)

    vertices /= np.linalg.norm(
        vertices,
        axis=1,
        keepdims=True
    )

    edges = [
        (0,1),(0,5),(0,7),(0,10),(0,11),
        (1,5),(1,7),(1,8),(1,9),
        (2,3),(2,4),(2,6),(2,10),(2,11),
        (3,4),(3,6),(3,8),(3,9),
        (4,5),(4,9),(4,11),
        (5,9),(5,11),
        (6,7),(6,8),(6,10),
        (7,8),(7,10),
        (8,9),(10,11)
    ]

    return vertices, edges


SHAPE_FUNCTIONS = {
    "Cube": cube,
    "Pyramid": pyramid,
    "Sphere": sphere,
    "Cylinder": cylinder,
    "Cone": cone,
    "Torus": torus,
    "Icosahedron": icosahedron,
}


# ============================================================
# ROTATION
# ============================================================

def rotation_matrix(ax, ay, az):

    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    rx = np.array([
        [1,0,0],
        [0,cx,-sx],
        [0,sx,cx]
    ])

    ry = np.array([
        [cy,0,sy],
        [0,1,0],
        [-sy,0,cy]
    ])

    rz = np.array([
        [cz,-sz,0],
        [sz,cz,0],
        [0,0,1]
    ])

    return rz @ ry @ rx


# ============================================================
# MEDIAPIPE
# ============================================================

class HandTracker:

    def __init__(self):

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

    def process(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        return self.detector.detect(image)

    def close(self):
        self.detector.close()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()
        self.tracker = HandTracker()

        self.shape_name = "Cube"

        self.scale = 100
        self.pos_x = 320
        self.pos_y = 240

        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0

        self.hand_count = 0
        self.fps = 0
        self.gesture = "Waiting for hands..."

        self.previous_time = time.perf_counter()

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(image, 1)

        height, width = image.shape[:2]

        result = self.tracker.process(image)

        hands = result.hand_landmarks or []

        self.hand_count = len(hands)


        # ----------------------------------------------------
        # ONE HAND
        # ----------------------------------------------------

        if len(hands) == 1:

            hand = hands[0]

            thumb = hand[THUMB_TIP]
            index = hand[INDEX_TIP]

            tx = int(thumb.x * width)
            ty = int(thumb.y * height)

            ix = int(index.x * width)
            iy = int(index.y * height)

            center_x = (tx + ix) / 2
            center_y = (ty + iy) / 2

            distance = math.hypot(
                ix - tx,
                iy - ty
            )

            target_scale = max(
                30,
                distance * 1.5
            )

            self.scale += (
                target_scale - self.scale
            ) * 0.25

            self.pos_x += (
                center_x - self.pos_x
            ) * 0.35

            self.pos_y += (
                center_y - self.pos_y
            ) * 0.35

            angle = math.atan2(
                iy - ty,
                ix - tx
            )

            self.angle_z += (
                angle - self.angle_z
            ) * 0.2

            self.gesture = (
                "Pinch • Scaling"
                if distance < 45
                else "Hand • Position / Rotate"
            )

            cv2.line(
                image,
                (tx, ty),
                (ix, iy),
                (50,230,150),
                3
            )


        # ----------------------------------------------------
        # TWO HANDS
        # ----------------------------------------------------

        elif len(hands) >= 2:

            p1 = hands[0][INDEX_TIP]
            p2 = hands[1][INDEX_TIP]

            x1, y1 = int(p1.x * width), int(p1.y * height)
            x2, y2 = int(p2.x * width), int(p2.y * height)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            distance = math.hypot(
                x2 - x1,
                y2 - y1
            )

            target_scale = max(
                40,
                distance / 1.5
            )

            self.scale += (
                target_scale - self.scale
            ) * 0.25

            self.pos_x += (
                center_x - self.pos_x
            ) * 0.35

            self.pos_y += (
                center_y - self.pos_y
            ) * 0.35

            self.gesture = (
                "Two Hands • Scale / Position"
            )

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (50,230,150),
                3
            )


        # ----------------------------------------------------
        # NO HANDS
        # ----------------------------------------------------

        else:

            self.gesture = (
                "Idle • Automatic Rotation"
            )

            self.angle_x += 0.015
            self.angle_y += 0.025
            self.angle_z += 0.005


        # ----------------------------------------------------
        # HAND LANDMARKS
        # ----------------------------------------------------

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        for hand in hands:

            points = []

            for lm in hand:

                point = (
                    int(lm.x * width),
                    int(lm.y * height)
                )

                points.append(point)

                cv2.circle(
                    image,
                    point,
                    3,
                    (0,255,180),
                    -1
                )

            for a, b in connections:

                cv2.line(
                    image,
                    points[a],
                    points[b],
                    (90,210,120),
                    2
                )


        # ----------------------------------------------------
        # 3D OBJECT
        # ----------------------------------------------------

        vertices, edges = SHAPE_FUNCTIONS[
            self.shape_name
        ]()

        rotation = rotation_matrix(
            self.angle_x,
            self.angle_y,
            self.angle_z
        )

        rotated = vertices @ rotation.T

        projected = [
            (
                int(v[0] * self.scale + self.pos_x),
                int(v[1] * self.scale + self.pos_y)
            )
            for v in rotated
        ]

        for a, b in edges:

            cv2.line(
                image,
                projected[a],
                projected[b],
                (255,120,40),
                2,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        now = time.perf_counter()
        dt = now - self.previous_time
        self.previous_time = now

        if dt > 0:

            current_fps = 1 / dt

            self.fps = (
                self.fps * 0.9
                + current_fps * 0.1
            )


        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (15,15),
            (300,125),
            (7,12,18),
            -1
        )

        cv2.putText(
            image,
            "HANDTRACKER 3D",
            (30,42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        cv2.putText(
            image,
            f"OBJECT  {self.shape_name.upper()}",
            (30,67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,150,80),
            1
        )

        cv2.putText(
            image,
            f"HANDS   {self.hand_count}",
            (30,88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70,230,170),
            1
        )

        cv2.putText(
            image,
            f"SCALE   {int(self.scale)}",
            (30,109),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190,200,210),
            1
        )

        cv2.putText(
            image,
            f"{self.fps:.1f} FPS",
            (width - 120,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60,220,255),
            2
        )

        cv2.putText(
            image,
            self.gesture,
            (25,height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230,230,230),
            1
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def stop(self):
        self.tracker.close()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">🖐️</div>
            <div class="sb-title">Control Center</div>
            <div class="sb-subtitle">HandTracker 3D AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🧊</span> 3D Object</div>',
        unsafe_allow_html=True,
    )

    selected_shape = st.selectbox(
        "3D Object",
        SHAPES,
        format_func=lambda s: f"{SHAPE_ICONS.get(s, '🔷')}  {s}",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🖐️</span> Gesture Controls</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-card">
            <p><b>Single Hand</b></p>
            <p>• Move hand → Position</p>
            <p>• Thumb + Index → Scale</p>
            <p>• Hand angle → Rotate</p>
        </div>
        <div class="sb-card">
            <p><b>Two Hands</b></p>
            <p>• Move apart → Increase</p>
            <p>• Move together → Decrease</p>
            <p>• Midpoint → Position</p>
        </div>
        <div class="sb-card">
            <p><b>No Hands</b></p>
            <p>• Automatic rotation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">📦</span> Available Objects</div>',
        unsafe_allow_html=True,
    )

    shapes_html = "".join(
        f'<div class="sb-shape-pill">{SHAPE_ICONS.get(shape, "🔷")} {shape}</div>'
        for shape in SHAPES
    )
    st.markdown(shapes_html, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">⚙️</span> System</div>',
        unsafe_allow_html=True,
    )

    for label in [
        "MediaPipe Hand Landmarker",
        "OpenCV + NumPy",
        "WebRTC Camera",
    ]:
        st.markdown(
            f'<div class="sb-status-chip">● {label}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# LIVE TRACKING
# ============================================================

st.markdown(
    '<div class="section-header"><span class="dot"></span> 🎥 Live Tracking Studio</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Allow camera access and move your hands in front "
    "of the camera to control the 3D object."
)

ctx = webrtc_streamer(
    key="hand-tracker-3d",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302",
                    "stun:stun3.l.google.com:19302",
                    "stun:stun4.l.google.com:19302",
                ]
            }
        ]
    },
    async_processing=True,
)

if ctx.video_processor is not None:
    ctx.video_processor.shape_name = selected_shape


# ============================================================
# METRICS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-header"><span class="dot"></span> 📊 System Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("🖐️ Max Hands", "2")

with c2:
    st.metric("🎯 Landmarks", "21")

with c3:
    st.metric("🧊 3D Objects", "7")

with c4:
    st.metric("⚡ Processing", "LIVE")

with c5:
    st.metric("🤖 Vision Engine", "AI")


# ============================================================
# TECHNOLOGY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-header"><span class="dot"></span> 🛠️ Technology Stack</div>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5 = st.columns(5)

tech_items = [
    ("🧠", "MediaPipe", "Hand landmark detection."),
    ("👁️", "OpenCV", "Video processing."),
    ("🔢", "NumPy", "3D calculations."),
    ("🌐", "Streamlit", "Application interface."),
    ("📹", "WebRTC", "Live camera stream."),
]

for col, (icon, name, desc) in zip([t1, t2, t3, t4, t5], tech_items):
    with col:
        st.markdown(
            f"""
            <div class="tech-card">
                <div class="tech-icon">{icon}</div>
                <b>{name}</b>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.expander("ℹ️ How does it work?"):

    st.write("""
    MediaPipe detects 21 landmarks for each hand.

    OpenCV and NumPy process the landmark positions to calculate
    hand position, finger distance and rotation.

    These values control the position, scale and rotation of
    the selected wireframe 3D object.

    When no hands are detected, the object automatically rotates.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-caption">🖐️ HandTracker 3D AI &nbsp;•&nbsp; '
    'OpenCV + MediaPipe + NumPy + WebRTC</div>',
    unsafe_allow_html=True,
)