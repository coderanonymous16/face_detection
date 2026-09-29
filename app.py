import tempfile
from random import randrange as r

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="Face Detection", page_icon="🙂", layout="centered")


@st.cache_resource
def load_cascade():
    # Ships with OpenCV, so no separate facedata.xml is needed
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


cascade = load_cascade()


def detect_faces(bgr, scale_factor, min_neighbors, only_first, color=None):
    """Core logic shared by image, camera and video modes."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if only_first and len(faces) > 0:
        faces = faces[:1]
    out = bgr.copy()
    for x, y, w, h in faces:
        c = color or (r(0, 256), r(0, 256), r(0, 256))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
    return out, len(faces)


def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------- Sidebar ----------------
st.title("Face Detection")
st.caption("Haar Cascade + OpenCV. Upload an image, use your camera, or try a video.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Detect", ["All faces", "Only first face"])
    scale_factor = st.slider("Scale factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.slider("Min neighbors", 1, 10, 3)

only_first = mode == "Only first face"

tab_live, tab_img, tab_cam, tab_vid = st.tabs(["Live", "Image", "Camera", "Video"])

# ---------------- Live (real time) ----------------
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.scale_factor = 1.1
        self.min_neighbors = 3
        self.only_first = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # fixed colour, so boxes do not flicker every frame
        out, _ = detect_faces(
            img, self.scale_factor, self.min_neighbors, self.only_first, (0, 0, 255)
        )
        return av.VideoFrame.from_ndarray(out, format="bgr24")


with tab_live:
    st.write("Click **START**, then allow camera access in your browser.")
    ctx = webrtc_streamer(
        key="live-face",
        video_processor_factory=FaceProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )
    if ctx.video_processor:  # sidebar settings update the live stream
        ctx.video_processor.scale_factor = scale_factor
        ctx.video_processor.min_neighbors = min_neighbors
        ctx.video_processor.only_first = only_first

# ---------------- Image ----------------
with tab_img:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if up:
        img = decode_image(up.read())
        if img is None:
            st.error("Could not read this image.")
        else:
            out, n = detect_faces(img, scale_factor, min_neighbors, only_first)
            st.image(out, channels="BGR", caption=f"Faces found: {n}")

# ---------------- Camera ----------------
with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        img = decode_image(shot.getvalue())
        out, n = detect_faces(img, scale_factor, min_neighbors, only_first)
        st.image(out, channels="BGR", caption=f"Faces found: {n}")

# ---------------- Video ----------------
with tab_vid:
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    n_frames = st.slider("Frames to preview", 3, 12, 6)
    if vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid.read())
            path = tmp.name

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            st.error("Could not read this video.")
        else:
            idxs = np.linspace(0, total - 1, n_frames, dtype=int)
            cols = st.columns(2)
            for i, idx in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                if w > 800:  # keep it fast
                    frame = cv2.resize(frame, (800, int(h * 800 / w)))
                out, n = detect_faces(frame, scale_factor, min_neighbors, only_first)
                cols[i % 2].image(out, channels="BGR", caption=f"Frame {idx} | faces: {n}")
        cap.release()
