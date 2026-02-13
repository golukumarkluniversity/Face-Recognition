"""
Face Recognizer Module
Handles face recognition and matching against the database.
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from .face_encoder import FaceEncoder


class FaceRecognizer:
    """Face recognition class for identifying known faces."""
    
    def __init__(self, data_dir: str = "data", tolerance: float = 0.6):
        """
        Initialize the face recognizer.
        
        Args:
            data_dir: Directory containing encodings file
            tolerance: Recognition tolerance (lower = stricter, higher = more lenient)
        """
        self.encoder = FaceEncoder(data_dir)
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.loaded = False
    
    def load_database(self) -> bool:
        """
        Load the face encodings database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.known_encodings, self.known_names = self.encoder.load_encodings()
            self.loaded = True
            print(f"Loaded {len(self.known_encodings)} face encodings from database")
            return True
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            return False
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return False
    
    def recognize_faces(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (face_locations, names, distances)
        """
        if not self.loaded:
            raise RuntimeError("Database not loaded. Call load_database() first.")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            return [], [], []
        
        # Encode detected faces
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        names = []
        distances = []
        
        # Compare each face with known faces
        for face_encoding in face_encodings:
            # Calculate distances to all known faces
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            # Check if the best match is within tolerance
            if best_distance <= self.tolerance:
                name = self.known_names[best_match_index]
                distance = best_distance
            else:
                name = "Unknown"
                distance = best_distance
            
            names.append(name)
            distances.append(distance)
        
        return face_locations, names, distances
    
    def recognize_from_image_file(self, image_path: str) -> bool:
        """
        Recognize faces from an image file and display results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return False
            
            print(f"Processing image: {image_path}")
            
            # Recognize faces
            face_locations, names, distances = self.recognize_faces(image)
            
            if len(face_locations) == 0:
                print("No faces detected in the image")
                return True
            
            print(f"\nFound {len(face_locations)} face(s):")
            
            # Display results
            for i, (name, distance) in enumerate(zip(names, distances)):
                confidence = (1 - distance) * 100
                print(f"  Face {i+1}: {name} (confidence: {confidence:.1f}%)")
            
            # Draw results on image
            output_image = image.copy()
            
            for (top, right, bottom, left), name, distance in zip(face_locations, names, distances):
                # Choose color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(output_image, (left, top), (right, bottom), color, 2)
                
                # Draw label
                confidence = (1 - distance) * 100
                label = f"{name} ({confidence:.1f}%)"
                cv2.rectangle(output_image, (left, top - 35), (right, top), color, cv2.FILLED)
                cv2.putText(output_image, label, (left + 6, top - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display image
            cv2.imshow("Face Recognition Result", output_image)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
            return False
    
    def recognize_from_webcam(self) -> None:
        """
        Real-time face recognition from webcam feed.
        """
        if not self.loaded:
            print("Error: Database not loaded. Call load_database() first.")
            return
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not access webcam")
            return
        
        print("\nStarting webcam recognition...")
        print("Press 'q' to quit")
        
        # Process every N frames for better performance
        process_every_n_frames = 2
        frame_count = 0
        
        # Store last recognition results
        last_face_locations = []
        last_names = []
        last_distances = []
        
        try:
            while True:
                # Capture frame
                ret, frame = video_capture.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    
                    # Recognize faces
                    face_locations, names, distances = self.recognize_faces(small_frame)
                    
                    # Scale back face locations to original size
                    last_face_locations = [(top*2, right*2, bottom*2, left*2) 
                                          for (top, right, bottom, left) in face_locations]
                    last_names = names
                    last_distances = distances
                
                # Draw results on frame
                display_frame = frame.copy()
                
                for (top, right, bottom, left), name, distance in zip(last_face_locations, last_names, last_distances):
                    # Choose color
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    confidence = (1 - distance) * 100
                    label = f"{name} ({confidence:.1f}%)"
                    cv2.rectangle(display_frame, (left, top - 35), (right, top), color, cv2.FILLED)
                    cv2.putText(display_frame, label, (left + 6, top - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Webcam Face Recognition - Press Q to quit', display_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Release resources
            video_capture.release()
            cv2.destroyAllWindows()
            print("\nWebcam recognition stopped")
