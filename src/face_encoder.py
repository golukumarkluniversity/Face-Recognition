"""
Face Encoder Module
Handles face encoding and database management for face recognition.
"""

import os
import pickle
import face_recognition
import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class FaceEncoder:
    """Face encoding and database management class."""
    
    def __init__(self, data_dir: str = "data", model: str = "small"):
        """
        Initialize the face encoder.
        
        Args:
            data_dir: Directory containing known_faces folder and encodings file
            model: Encoding model ('small' for faster, 'large' for more accurate)
        """
        self.data_dir = Path(data_dir)
        self.known_faces_dir = self.data_dir / "known_faces"
        self.encodings_file = self.data_dir / "encodings.pkl"
        self.model = model
        
        # Create directories if they don't exist
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
    
    def encode_faces_from_directory(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Encode all faces from the known_faces directory.
        
        Directory structure should be:
        known_faces/
            person_name_1/
                image1.jpg
                image2.jpg
            person_name_2/
                image1.jpg
        
        Returns:
            Tuple of (encodings, names) lists
        """
        encodings = []
        names = []
        
        # Check if directory exists
        if not self.known_faces_dir.exists():
            raise FileNotFoundError(f"Directory {self.known_faces_dir} does not exist")
        
        # Iterate through each person's directory
        person_dirs = [d for d in self.known_faces_dir.iterdir() if d.is_dir()]
        
        if not person_dirs:
            print(f"Warning: No subdirectories found in {self.known_faces_dir}")
            return encodings, names
        
        total_images = 0
        successful_encodings = 0
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            print(f"\nProcessing images for: {person_name}")
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(person_dir.glob(ext))
            
            if not image_files:
                print(f"  Warning: No images found for {person_name}")
                continue
            
            # Process each image
            for image_path in image_files:
                total_images += 1
                print(f"  Processing: {image_path.name}... ", end="")
                
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print("Failed to load image")
                        continue
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Find faces in the image
                    face_locations = face_recognition.face_locations(rgb_image, model="hog")
                    
                    if len(face_locations) == 0:
                        print("No face detected")
                        continue
                    
                    if len(face_locations) > 1:
                        print(f"Multiple faces detected ({len(face_locations)}), using first face")
                    
                    # Encode the first face found
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations, model=self.model)
                    
                    if face_encodings:
                        encodings.append(face_encodings[0])
                        names.append(person_name)
                        successful_encodings += 1
                        print("Success")
                    else:
                        print("Failed to encode")
                        
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        print(f"\n{'='*50}")
        print(f"Total images processed: {total_images}")
        print(f"Successful encodings: {successful_encodings}")
        print(f"{'='*50}\n")
        
        return encodings, names
    
    def save_encodings(self, encodings: List[np.ndarray], names: List[str]) -> None:
        """
        Save face encodings to a pickle file.
        
        Args:
            encodings: List of face encodings
            names: List of corresponding names
        """
        if len(encodings) == 0:
            print("Warning: No encodings to save")
            return
        
        data = {
            "encodings": encodings,
            "names": names
        }
        
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Encodings saved successfully to {self.encodings_file}")
            print(f"Total faces registered: {len(names)}")
        except Exception as e:
            print(f"Error saving encodings: {str(e)}")
            raise
    
    def load_encodings(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load face encodings from pickle file.
        
        Returns:
            Tuple of (encodings, names) lists
        """
        if not self.encodings_file.exists():
            raise FileNotFoundError(f"Encodings file {self.encodings_file} not found. Please register faces first.")
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            
            encodings = data.get("encodings", [])
            names = data.get("names", [])
            
            return encodings, names
        except Exception as e:
            print(f"Error loading encodings: {str(e)}")
            raise
    
    def register_faces(self) -> bool:
        """
        Complete face registration workflow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Starting face registration process...")
            print(f"Scanning directory: {self.known_faces_dir}")
            
            # Encode faces
            encodings, names = self.encode_faces_from_directory()
            
            if len(encodings) == 0:
                print("\nNo faces were encoded. Please check:")
                print(f"1. Images are placed in subdirectories under {self.known_faces_dir}")
                print("2. Images contain clear, visible faces")
                print("3. Image formats are supported (jpg, jpeg, png)")
                return False
            
            # Save encodings
            self.save_encodings(encodings, names)
            
            return True
            
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            return False
    
    def get_registered_names(self) -> List[str]:
        """
        Get list of all registered face names.
        
        Returns:
            List of unique names
        """
        try:
            _, names = self.load_encodings()
            return sorted(list(set(names)))
        except FileNotFoundError:
            return []
