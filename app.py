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

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9


# ============================================================
# SIMPLE CSS
# ============================================================

st.markdown("""
<style>

#MainMenu, footer {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, #ff6b3515, transparent 30%),
        radial-gradient(circle at 90% 10%, #00d4ff10, transparent 30%),
        #070b10;
    color: white;
}

[data-testid="stSidebar"] {
    background: #090e14;
    border-right: 1px solid #202b36;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    margin-bottom: 25px;
    background: linear-gradient(135deg, #ff6b3520, #00d4ff08);
    border: 1px solid #ffffff12;
}

.hero h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 900;
}

.hero p {
    color: #9aa7b5;
    max-width: 850px;
    font-size: 15px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 20px;
    background: #00d4ff12;
    border: 1px solid #00d4ff30;
    color: #62ddff;
    font-size: 11px;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background: #101720;
    border: 1px solid #202c38;
    padding: 15px;
    border-radius: 15px;
}

[data-testid="stWebRtcStreamer"] {
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid #263441 !important;
}

.stButton button {
    border-radius: 10px;
    background: #111922;
    border: 1px solid #293643;
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

    st.title("🎛️ Control Center")

    st.caption(
        "Configure your 3D tracking experience."
    )

    selected_shape = st.selectbox(
        "3D Object",
        SHAPES
    )

    st.divider()

    st.subheader("🖐️ Gesture Controls")

    st.markdown("""
    **Single Hand**
    
    • Move hand → Position  
    • Thumb + Index → Scale  
    • Hand angle → Rotate  

    **Two Hands**

    • Move apart → Increase  
    • Move together → Decrease  
    • Midpoint → Position  

    **No Hands**

    • Automatic rotation
    """)

    st.divider()

    st.subheader("📦 Available Objects")

    for shape in SHAPES:
        st.write(f"• {shape}")

    st.divider()

    st.subheader("⚙️ System")

    st.success("MediaPipe Hand Landmarker")
    st.success("OpenCV + NumPy")
    st.success("WebRTC Camera")


# ============================================================
# LIVE TRACKING
# ============================================================

st.subheader("🎥 Live Tracking Studio")

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
    async_processing=True,
)


if ctx.video_processor:

    ctx.video_processor.shape_name = selected_shape


# ============================================================
# METRICS
# ============================================================

st.subheader("📊 System Overview")

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

st.subheader("🛠️ Technology Stack")

t1, t2, t3, t4, t5 = st.columns(5)

with t1:
    st.info("🧠 **MediaPipe**\n\nHand landmark detection.")

with t2:
    st.info("👁️ **OpenCV**\n\nVideo processing.")

with t3:
    st.info("🔢 **NumPy**\n\n3D calculations.")

with t4:
    st.info("🌐 **Streamlit**\n\nApplication interface.")

with t5:
    st.info("📹 **WebRTC**\n\nLive camera stream.")


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

st.caption(
    "🖐️ HandTracker 3D AI • "
    "OpenCV + MediaPipe + NumPy + WebRTC"
)