import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

# Memory for tracking lips
prev_lip_positions_list = []
face_colors = {}

# Lip landmark definitions
lip_indexes = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 80, 191
]
key_indexes = [13, 14, 17, 0, 61, 291]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape
    cropped_lips = []  # Store each cropped lips image

    if results.multi_face_landmarks:
        # Reset per-frame memory
        new_prev_lip_positions = []

        for face_id, landmarks in enumerate(results.multi_face_landmarks):
            # Get all lip landmarks
            current_positions = []
            for i in lip_indexes:
                x = int(landmarks.landmark[i].x * w)
                y = int(landmarks.landmark[i].y * h)
                current_positions.append((x, y))

            # Get key points for motion detection
            current_key_positions = []
            for i in key_indexes:
                x = int(landmarks.landmark[i].x * w)
                y = int(landmarks.landmark[i].y * h)
                current_key_positions.append((x, y))

            # Detect movement
            is_moving = False
            if face_id < len(prev_lip_positions_list):
                key_deltas = [
                    np.linalg.norm(np.array(curr) - np.array(prev))
                    for curr, prev in zip(current_key_positions, prev_lip_positions_list[face_id])
                ]
                avg_delta = np.mean(key_deltas)
                if avg_delta > 0.5:
                    is_moving = True

            # Assign color for this face if not already done
            if face_id not in face_colors:
                face_colors[face_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

            color = face_colors[face_id] if is_moving else (0, 0, 255)  # red if still

            # Draw lips on original frame
            for x, y in current_positions:
                cv2.circle(frame, (x, y), 2, color, -1)

            # Save current key points
            new_prev_lip_positions.append(current_key_positions)

            # Crop lips
            x_coords = [pt[0] for pt in current_positions]
            y_coords = [pt[1] for pt in current_positions]

            x_min = max(min(x_coords) - 10, 0)
            y_min = max(min(y_coords) - 10, 0)
            x_max = min(max(x_coords) + 10, w)
            y_max = min(max(y_coords) + 10, h)

            lips_crop = frame[y_min:y_max, x_min:x_max]
            cropped_lips.append(lips_crop)

        # Update memory for next frame
        prev_lip_positions_list = new_prev_lip_positions

    # Combine all cropped lips into one wide image
    if cropped_lips:
        lips_strip = cv2.hconcat([
            cv2.resize(lip, (100, 100)) for lip in cropped_lips
        ])
        cv2.imshow("All Lips Cropped", lips_strip)

    # Show original frame
    cv2.imshow("Lips Movement Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()