"""
Face Detector Module
Handles face detection in images using OpenCV and face_recognition library.
"""

import cv2
import face_recognition
from typing import List, Tuple, Optional
import numpy as np


class FaceDetector:
    """Face detection class using HOG or CNN-based detection."""
    
    def __init__(self, model: str = "hog"):
        """
        Initialize the face detector.
        
        Args:
            model: Detection model to use ('hog' for CPU or 'cnn' for GPU)
        """
        if model not in ["hog", "cnn"]:
            raise ValueError("Model must be 'hog' or 'cnn'")
        self.model = model
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model=self.model)
        
        return face_locations
    
    def draw_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                   labels: Optional[List[str]] = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            face_locations: List of face locations
            labels: Optional list of labels for each face
            color: Color for the bounding box (BGR format)
            
        Returns:
            Image with drawn bounding boxes
        """
        output_image = image.copy()
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw rectangle
            cv2.rectangle(output_image, (left, top), (right, bottom), color, 2)
            
            # Draw label if provided
            if labels and i < len(labels):
                label = labels[i]
                # Draw label background
                cv2.rectangle(output_image, (left, top - 35), (right, top), color, cv2.FILLED)
                # Draw label text
                cv2.putText(output_image, label, (left + 6, top - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return output_image
    
    def detect_single_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect a single face in an image.
        
        Args:
            image: Input image
            
        Returns:
            Face location if exactly one face is found, None otherwise
        """
        face_locations = self.detect_faces(image)
        
        if len(face_locations) == 1:
            return face_locations[0]
        return None
