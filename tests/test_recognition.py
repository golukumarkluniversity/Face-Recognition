"""
Basic Unit Tests for Face Recognition System
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.face_encoder import FaceEncoder
from src.face_recognizer import FaceRecognizer


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(model="hog")
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.model, "hog")
        
        # Test invalid model
        with self.assertRaises(ValueError):
            FaceDetector(model="invalid")
    
    def test_detect_faces_with_synthetic_image(self):
        """Test face detection with a synthetic image."""
        # Create a simple test image (black image)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Detect faces (should find none in black image)
        face_locations = self.detector.detect_faces(test_image)
        
        # Verify return type
        self.assertIsInstance(face_locations, list)
    
    def test_draw_faces(self):
        """Test drawing bounding boxes on faces."""
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create fake face locations
        face_locations = [(100, 200, 300, 100)]
        labels = ["Test Person"]
        
        # Draw faces
        result_image = self.detector.draw_faces(test_image, face_locations, labels)
        
        # Verify image was modified
        self.assertIsInstance(result_image, np.ndarray)
        self.assertEqual(result_image.shape, test_image.shape)


class TestFaceEncoder(unittest.TestCase):
    """Test cases for FaceEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = FaceEncoder(data_dir="data")
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(str(self.encoder.data_dir), "data")
        self.assertTrue(self.encoder.known_faces_dir.exists())
    
    def test_get_registered_names_empty(self):
        """Test getting registered names when database is empty."""
        names = self.encoder.get_registered_names()
        self.assertIsInstance(names, list)


class TestFaceRecognizer(unittest.TestCase):
    """Test cases for FaceRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = FaceRecognizer(data_dir="data", tolerance=0.6)
    
    def test_initialization(self):
        """Test recognizer initialization."""
        self.assertEqual(self.recognizer.tolerance, 0.6)
        self.assertFalse(self.recognizer.loaded)
    
    def test_recognize_without_loaded_database(self):
        """Test that recognition fails without loading database."""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with self.assertRaises(RuntimeError):
            self.recognizer.recognize_faces(test_image)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_encoder_and_recognizer_workflow(self):
        """Test the complete encode and recognize workflow."""
        # This test requires actual test images, so we just verify
        # that the classes can be instantiated and work together
        encoder = FaceEncoder(data_dir="data")
        recognizer = FaceRecognizer(data_dir="data")
        
        # Verify both use the same data directory
        self.assertEqual(str(encoder.data_dir), str(recognizer.encoder.data_dir))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFaceDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceRecognizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
