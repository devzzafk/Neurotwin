import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

def calculate_focus_score(blink_rate, movement):
    focus = max(0, 100 - (blink_rate * 2 + movement * 50))
    return min(100, focus)

def run_tracker(duration=30):
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh()
    
    start_time = time.time()
    blink_count = 0
    movement_total = 0
    prev_center = None

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye = landmarks[159].y
            right_eye = landmarks[386].y
            eye_diff = abs(left_eye - right_eye)

            if eye_diff < 0.005:
                blink_count += 1

            nose = landmarks[1]
            center = np.array([nose.x, nose.y])

            if prev_center is not None:
                movement_total += np.linalg.norm(center - prev_center)

            prev_center = center

    cap.release()

    blink_rate = blink_count / duration
    focus_score = calculate_focus_score(blink_rate, movement_total)

    return {
        "blink_rate": blink_rate,
        "movement": movement_total,
        "focus_score": focus_score
    }
