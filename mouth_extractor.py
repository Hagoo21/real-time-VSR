import cv2
import mediapipe as mp
import numpy as np
import random

class MouthExtractor:
    def __init__(self, max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize the mouth extractor with configurable parameters"""
        # Initialize mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Memory for tracking lips
        self.prev_lip_positions_list = []
        self.face_colors = {}
        
        # Lip landmark definitions
        self.lip_indexes = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 80, 191
        ]
        self.key_indexes = [13, 14, 17, 0, 61, 291]
    
    def extract_mouth_regions_with_motion(self, frame):
        """Extract mouth regions from frame with motion detection and face tracking"""
        if frame is None:
            return [], frame
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        h, w, _ = frame.shape
        cropped_lips = []
        display_frame = frame.copy()
        
        if results.multi_face_landmarks:
            # Reset per-frame memory
            new_prev_lip_positions = []
            
            for face_id, landmarks in enumerate(results.multi_face_landmarks):
                # Get all lip landmarks
                current_positions = []
                for i in self.lip_indexes:
                    x = int(landmarks.landmark[i].x * w)
                    y = int(landmarks.landmark[i].y * h)
                    current_positions.append((x, y))
                
                # Get key points for motion detection
                current_key_positions = []
                for i in self.key_indexes:
                    x = int(landmarks.landmark[i].x * w)
                    y = int(landmarks.landmark[i].y * h)
                    current_key_positions.append((x, y))
                
                # Detect movement
                is_moving = False
                if face_id < len(self.prev_lip_positions_list):
                    key_deltas = [
                        np.linalg.norm(np.array(curr) - np.array(prev))
                        for curr, prev in zip(current_key_positions, self.prev_lip_positions_list[face_id])
                    ]
                    avg_delta = np.mean(key_deltas)
                    if avg_delta > 0.5:
                        is_moving = True
                
                # Assign color for this face if not already done
                if face_id not in self.face_colors:
                    self.face_colors[face_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                
                color = self.face_colors[face_id] if is_moving else (0, 0, 255)  # red if still
                
                # Draw lips on display frame
                for x, y in current_positions:
                    cv2.circle(display_frame, (x, y), 2, color, -1)
                
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
                # Only add non-empty crops
                if lips_crop.size > 0 and lips_crop.shape[0] > 0 and lips_crop.shape[1] > 0:
                    cropped_lips.append(lips_crop)
            
            # Update memory for next frame
            self.prev_lip_positions_list = new_prev_lip_positions
        
        return cropped_lips, display_frame
    
    def extract_mouth_regions(self, frame):
        """Extract mouth regions from frame (simple version without motion tracking)"""
        cropped_lips, _ = self.extract_mouth_regions_with_motion(frame)
        return cropped_lips
    
    def create_mouth_strip(self, mouth_regions, size=(100, 100)):
        """Create a horizontal strip of mouth regions"""
        if not mouth_regions:
            return None
        
        # Filter out any empty images and resize valid ones
        valid_lips = []
        for lip in mouth_regions:
            if lip.size > 0 and lip.shape[0] > 0 and lip.shape[1] > 0:
                valid_lips.append(cv2.resize(lip, size))
        
        if valid_lips:
            return cv2.hconcat(valid_lips)
        return None

def main():
    """Main function for standalone mouth extractor demo"""
    # Initialize mouth extractor
    mouth_extractor = MouthExtractor()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract mouth regions with motion detection
            cropped_lips, display_frame = mouth_extractor.extract_mouth_regions_with_motion(frame)
            
            # Create and show mouth strip
            lips_strip = mouth_extractor.create_mouth_strip(cropped_lips)
            if lips_strip is not None:
                cv2.imshow("All Lips Cropped", lips_strip)
            
            # Show original frame with lip landmarks
            cv2.imshow("Lips Movement Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()