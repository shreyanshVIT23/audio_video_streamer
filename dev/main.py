"""
Smart Face Recognition Pipeline
Uses lightweight models for detection and heavier models for identification
"""

import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import defaultdict
from pathlib import Path

class SmartFaceRecognitionPipeline:
    def __init__(
        self,
        detection_backend='opencv',  # Lightweight detector
        recognition_model='Facenet512',  # Heavy recognition model
        detection_confidence=0.7,
        recognition_threshold=0.4,
        known_faces_dir='known_faces',
        skip_frames=2  # Process every Nth frame for efficiency
    ):
        """
        Initialize the pipeline with configurable models.
        
        Args:
            detection_backend: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet'
            recognition_model: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'ArcFace', 'Dlib', 'SFace'
            detection_confidence: Confidence threshold for face detection
            recognition_threshold: Distance threshold for face recognition
            known_faces_dir: Directory containing known face images
            skip_frames: Process every Nth frame to improve performance
        """
        self.detection_backend = detection_backend
        self.recognition_model = recognition_model
        self.detection_confidence = detection_confidence
        self.recognition_threshold = recognition_threshold
        self.known_faces_dir = known_faces_dir
        self.skip_frames = skip_frames
        
        # Cache for recognized faces to avoid redundant processing
        self.face_cache = {}
        self.cache_ttl = 30  # Cache results for 30 frames
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'detection_time': [],
            'recognition_time': []
        }
        
        print(f"Pipeline initialized:")
        print(f"  Detection: {detection_backend}")
        print(f"  Recognition: {recognition_model}")
        print(f"  Known faces directory: {known_faces_dir}")
    
    def _hash_face_region(self, face_bbox):
        """Create a simple hash for face region to use as cache key."""
        x, y, w, h = face_bbox
        return f"{x}_{y}_{w}_{h}"
    
    def detect_faces(self, frame):
        """
        Step 1: Use lightweight model to detect if faces are present.
        Returns list of face bounding boxes.
        """
        start_time = time.time()
        
        try:
            # Use DeepFace's extract_faces with lightweight backend
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detection_backend,
                enforce_detection=False,
                align=True
            )
            
            detection_time = time.time() - start_time
            self.stats['detection_time'].append(detection_time)
            
            if faces:
                self.stats['faces_detected'] += len(faces)
                return faces
            
            return []
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def recognize_face(self, frame, face_region):
        """
        Step 2: Use heavy model to identify who the person is.
        Returns the identity and confidence.
        """
        start_time = time.time()
        
        try:
            # Use DeepFace.find to search in known faces database
            result = DeepFace.find(
                img_path=face_region,
                db_path=self.known_faces_dir,
                model_name=self.recognition_model,
                enforce_detection=False,
                detector_backend=self.detection_backend,
                silent=True
            )
            
            recognition_time = time.time() - start_time
            self.stats['recognition_time'].append(recognition_time)
            
            # Check if any match found
            if len(result) > 0 and len(result[0]) > 0:
                top_match = result[0].iloc[0]
                distance = top_match['distance']
                
                if distance < self.recognition_threshold:
                    # Extract name from file path
                    identity_path = top_match['identity']
                    identity = Path(identity_path).stem
                    confidence = 1 - distance  # Convert distance to confidence
                    
                    self.stats['faces_recognized'] += 1
                    return identity, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process entire video with the smart pipeline.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            display: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                annotated_frame = frame.copy()
                
                # Skip frames for performance (process every Nth frame)
                if frame_count % self.skip_frames != 0:
                    if writer:
                        writer.write(annotated_frame)
                    continue
                
                self.stats['frames_processed'] += 1
                
                # Step 1: Lightweight detection
                faces = self.detect_faces(frame)
                
                # Step 2: Heavy recognition for each detected face
                for face in faces:
                    # Get face coordinates
                    facial_area = face['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Check cache first
                    face_hash = self._hash_face_region((x, y, w, h))
                    
                    if face_hash in self.face_cache:
                        cache_entry = self.face_cache[face_hash]
                        if cache_entry['age'] < self.cache_ttl:
                            identity, confidence = cache_entry['identity'], cache_entry['confidence']
                            cache_entry['age'] += 1
                        else:
                            # Cache expired, re-recognize
                            identity, confidence = self.recognize_face(frame, face_img)
                            self.face_cache[face_hash] = {'identity': identity, 'confidence': confidence, 'age': 0}
                    else:
                        # Not in cache, recognize
                        identity, confidence = self.recognize_face(frame, face_img)
                        self.face_cache[face_hash] = {'identity': identity, 'confidence': confidence, 'age': 0}
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Add label with identity and confidence
                    label = f"{identity}: {confidence:.2f}" if identity != "Unknown" else "Unknown"
                    cv2.putText(annotated_frame, label, (x, y-10), 
                               cv2.FontFace.SIMPLEX, 0.6, color, 2)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Faces: {len(faces)}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                           cv2.FontFace.SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                if display:
                    cv2.imshow('Smart Face Recognition Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            self._print_statistics()
    
    def process_webcam(self, camera_index=0):
        """
        Process live webcam feed.
        
        Args:
            camera_index: Camera device index (usually 0 for default webcam)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("\nProcessing webcam feed (press 'q' to quit)...")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % self.skip_frames != 0:
                    cv2.imshow('Smart Face Recognition Pipeline', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                self.stats['frames_processed'] += 1
                
                # Step 1: Lightweight detection
                faces = self.detect_faces(frame)
                
                # Step 2: Heavy recognition
                for face in faces:
                    facial_area = face['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    face_img = frame[y:y+h, x:x+w]
                    
                    identity, confidence = self.recognize_face(frame, face_img)
                    
                    # Draw results
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{identity}: {confidence:.2f}" if identity != "Unknown" else "Unknown"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FontFace.SIMPLEX, 0.6, color, 2)
                
                # Display
                cv2.imshow('Smart Face Recognition Pipeline', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_statistics()
    
    def _print_statistics(self):
        """Print pipeline performance statistics."""
        print("\n" + "="*50)
        print("Pipeline Statistics")
        print("="*50)
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Faces detected: {self.stats['faces_detected']}")
        print(f"Faces recognized: {self.stats['faces_recognized']}")
        
        if self.stats['detection_time']:
            avg_detection = np.mean(self.stats['detection_time'])
            print(f"Avg detection time: {avg_detection*1000:.2f}ms")
        
        if self.stats['recognition_time']:
            avg_recognition = np.mean(self.stats['recognition_time'])
            print(f"Avg recognition time: {avg_recognition*1000:.2f}ms")
        
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with lightweight detector and heavy recognizer
    pipeline = SmartFaceRecognitionPipeline(
        detection_backend='opencv',      # Fast: opencv, yunet, ssd
        recognition_model='Facenet512',  # Accurate: Facenet512, ArcFace, VGG-Face
        detection_confidence=0.7,
        recognition_threshold=0.4,
        known_faces_dir='known_faces',   # Directory with known face images
        skip_frames=2                     # Process every 2nd frame
    )
    
    # Process a video file
    # pipeline.process_video(
    #     video_path='input_video.mp4',
    #     output_path='output_video.mp4',
    #     display=True
    # )
    
    # Or process webcam feed
    # pipeline.process_webcam(camera_index=0)
    
    print("\nSetup complete! Uncomment the desired processing method above.")
    print("\nDirectory structure needed:")
    print("known_faces/")
    print("  ├── person1.jpg")
    print("  ├── person2.jpg")
    print("  └── person3.jpg")
