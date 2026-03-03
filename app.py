import streamlit as st
import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Self-Awareness AI", layout="wide")

st.title("🧠 Self-Awareness AI")
st.write("Real-time Micro-Behavior Analyzer")

mp_face_mesh = mp.solutions.face_mesh


class BehaviorAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.prev_center = None
        self.movement_score = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            nose = landmarks[1]
            center = np.array([nose.x, nose.y])

            if self.prev_center is not None:
                movement = np.linalg.norm(center - self.prev_center)
                self.movement_score += movement

            self.prev_center = center

            stability = max(0, 100 - self.movement_score * 200)
            engagement = min(100, abs(nose.z) * 300)

            cv2.putText(img, f"Stability: {int(stability)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.putText(img, f"Engagement: {int(engagement)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

        return img


webrtc_streamer(
    key="behavior",
    video_transformer_factory=BehaviorAnalyzer
)
