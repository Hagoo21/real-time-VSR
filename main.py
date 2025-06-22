import torch
import numpy as np
import cv2
import os
import sys
import tempfile
import time

# Import mouth extraction components for display purposes
import mediapipe as mp

class MouthExtractor:
    def __init__(self):
        # Initialize mediapipe face mesh for display purposes
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Lip landmark definitions
        self.lip_indexes = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 80, 191
        ]
    
    def extract_mouth_region(self, frame):
        """Extract mouth region from frame for display purposes"""
        if frame is None:
            return None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0]
        
        # Get lip landmarks
        lip_points = []
        for i in self.lip_indexes:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            lip_points.append((x, y))
        
        # Calculate bounding box for lips with padding
        x_coords = [pt[0] for pt in lip_points]
        y_coords = [pt[1] for pt in lip_points]
        
        x_min = max(min(x_coords) - 20, 0)
        y_min = max(min(y_coords) - 20, 0)
        x_max = min(max(x_coords) + 20, w)
        y_max = min(max(y_coords) + 20, h)
        
        # Extract mouth region for display
        mouth_crop = frame[y_min:y_max, x_min:x_max]
        if mouth_crop.size == 0:
            return None
            
        return mouth_crop

def get_camera(index=0):
    """Get camera capture object"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")
    return cap

def initialize_chaplin():
    """Initialize CHAPLIN by changing to chaplin directory and loading the model"""
    # Save current directory
    original_dir = os.getcwd()
    chaplin_dir = os.path.join(original_dir, 'chaplin')
    
    try:
        # Change to chaplin directory
        os.chdir(chaplin_dir)
        
        # Add chaplin to Python path
        if chaplin_dir not in sys.path:
            sys.path.insert(0, chaplin_dir)
        
        # Import CHAPLIN components
        from pipelines.pipeline import InferencePipeline
        
        # Initialize with proper config path (relative to chaplin directory)
        config_path = "configs/LRS3_V_WER19.1.ini"
        
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return None, original_dir
        
        # Initialize CHAPLIN pipeline
        vsr_model = InferencePipeline(
            config_path,
            detector="mediapipe",
            face_track=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        print("CHAPLIN model loaded successfully!")
        return vsr_model, original_dir
        
    except Exception as e:
        print(f"Error loading CHAPLIN model: {e}")
        os.chdir(original_dir)  # Restore directory on error
        return None, original_dir

def save_video_for_chaplin(frames, output_path, fps=25):
    """Save full video frames for CHAPLIN processing"""
    if not frames:
        return False
    
    # Get dimensions from first frame
    h, w = frames[0].shape[:2]
    
    # Use mp4v codec - CHAPLIN expects color video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), True)  # isColor=True
    
    for frame in frames:
        # Ensure frame is in color format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()
    return True

class ChaplinLipReader:
    def __init__(self):
        self.vsr_model, self.original_dir = initialize_chaplin()
        self.mouth_extractor = MouthExtractor()
        self.recording = False
        self.frame_buffer = []
        self.fps = 25
        self.min_frames = 50  # Minimum frames for processing (~2 seconds)
        
        # Video recording parameters
        self.frame_compression = 80  # Higher quality for better face detection
        
    def process_sequence(self):
        """Process the current frame buffer with CHAPLIN"""
        if not self.vsr_model or len(self.frame_buffer) < self.min_frames:
            return None
            
        try:
            # Create temporary video file in the chaplin directory
            temp_video_path = f"temp_sequence_{int(time.time() * 1000)}.mp4"
            
            # Save full frames as video (not just mouth regions)
            if save_video_for_chaplin(self.frame_buffer, temp_video_path, self.fps):
                print(f"Processing {len(self.frame_buffer)} frames...")
                
                # Run CHAPLIN inference
                transcript = self.vsr_model(temp_video_path)
                
                # Clean up temporary file
                try:
                    os.remove(temp_video_path)
                except:
                    pass
                
                return transcript
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
        return None
    
    def run(self):
        """Main loop for lip reading"""
        cap = get_camera(0)
        
        if self.vsr_model:
            print("CHAPLIN Lip Reader loaded!")
            print("Controls:")
            print("  SPACE - Start/Stop recording")
            print("  ENTER - Process current recording")
            print("  Q - Quit")
        else:
            print("Running in mouth extraction demo mode only")
            print("  Q - Quit")
        
        last_frame_time = time.time()
        frame_interval = 1.0 / self.fps
        
        try:
            while True:
                current_time = time.time()
                
                # Maintain consistent frame rate
                if current_time - last_frame_time >= frame_interval:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Apply compression but keep quality high enough for face detection
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                    
                    # Store full frames for CHAPLIN processing
                    if self.recording and self.vsr_model:
                        self.frame_buffer.append(compressed_frame.copy())
                    
                    # Extract mouth region for display purposes only
                    mouth_region = self.mouth_extractor.extract_mouth_region(compressed_frame)
                    
                    # Create display frame
                    display_frame = compressed_frame.copy()
                    
                    # Show mouth region if detected
                    if mouth_region is not None:
                        mouth_display = cv2.resize(mouth_region, (150, 150))
                        display_frame[10:160, 10:160] = mouth_display
                    
                    # Show recording indicator
                    if self.recording:
                        cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(display_frame, "REC", (display_frame.shape[1] - 50, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Show buffer info
                    if self.vsr_model:
                        buffer_text = f"Frames: {len(self.frame_buffer)}"
                        status_text = "Recording" if self.recording else "Ready"
                        cv2.putText(display_frame, buffer_text, (10, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, status_text, (10, 230), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Mouth extraction only", (10, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow("CHAPLIN Lip Reader", display_frame)
                    last_frame_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and self.vsr_model:  # Space to toggle recording
                    if self.recording:
                        self.recording = False
                        print(f"Recording stopped. Captured {len(self.frame_buffer)} frames.")
                    else:
                        self.recording = True
                        self.frame_buffer = []  # Clear buffer
                        print("Recording started...")
                elif key == 13 and self.vsr_model and not self.recording:  # Enter to process
                    if len(self.frame_buffer) >= self.min_frames:
                        print("Processing sequence...")
                        transcript = self.process_sequence()
                        if transcript:
                            print(f"🧠 Transcribed: '{transcript}'")
                            
                            # Show result on display
                            result_frame = display_frame.copy()
                            cv2.putText(result_frame, f"Result: {transcript}", (10, 260), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.imshow("CHAPLIN Lip Reader", result_frame)
                            cv2.waitKey(3000)  # Show for 3 seconds
                        else:
                            print("Failed to process sequence")
                        
                        self.frame_buffer = []  # Clear buffer after processing
                    else:
                        print(f"Need at least {self.min_frames} frames. Current: {len(self.frame_buffer)}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Restore original directory
            os.chdir(self.original_dir)
            print("Lip reading session ended.")

def main():
    """Main function"""
    lip_reader = ChaplinLipReader()
    lip_reader.run()

if __name__ == "__main__":
    main()
