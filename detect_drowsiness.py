import cv2
import mediapipe as mp
import numpy as np
import winsound              # for alarm sound
from tensorflow.keras.models import load_model

# If you still want to use the CNN model later, it's loaded here (not used in EAR logic)
model = load_model("model/drowsiness_model.h5")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points, landmarks):
    eye = np.array([(landmarks[p].x, landmarks[p].y) for p in eye_points])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.23     # smaller = more strict
CLOSED_FRAMES = 0
LIMIT = 15               # number of frames -> drowsy

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = "AWAKE"
    color = (0, 255, 0)   # green by default

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(LEFT_EYE, mesh_points)
        right_ear = eye_aspect_ratio(RIGHT_EYE, mesh_points)
        ear = (left_ear + right_ear) / 2.0

        # Check if eyes look closed
        if ear < EAR_THRESHOLD:
            CLOSED_FRAMES += 1
        else:
            CLOSED_FRAMES = 0

        # If eyes closed long enough -> DROWSY
        if CLOSED_FRAMES > LIMIT:
            status_text = "DROWSY!"
            color = (0, 0, 255)  # red
            winsound.Beep(2500, 800)  # frequency, duration (ms)

    # 🔹 ALWAYS draw status on the frame
    cv2.putText(
        frame,
        status_text,
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        color,
        4
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
