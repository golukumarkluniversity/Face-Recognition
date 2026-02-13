"""
Main Application
Interactive CLI for the Face Recognition System.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_encoder import FaceEncoder
from src.face_recognizer import FaceRecognizer


def print_banner():
    """Print application banner."""
    print("\n" + "="*60)
    print("          FACE RECOGNITION SYSTEM")
    print("="*60 + "\n")


def print_menu():
    """Print main menu options."""
    print("\nMain Menu:")
    print("-" * 40)
    print("1. Register new faces")
    print("2. Recognize faces from image file")
    print("3. Real-time webcam recognition")
    print("4. List all registered faces")
    print("5. Exit")
    print("-" * 40)


def register_faces():
    """Register new faces from known_faces directory."""
    print("\n" + "="*60)
    print("FACE REGISTRATION")
    print("="*60)
    print("\nInstructions:")
    print("1. Create a subdirectory for each person in 'data/known_faces/'")
    print("2. Place one or more photos of each person in their directory")
    print("3. Supported formats: jpg, jpeg, png")
    print("\nExample structure:")
    print("  data/known_faces/")
    print("    ├── John/")
    print("    │   ├── photo1.jpg")
    print("    │   └── photo2.jpg")
    print("    └── Jane/")
    print("        └── photo1.jpg")
    print()
    
    encoder = FaceEncoder()
    success = encoder.register_faces()
    
    if success:
        print("\n✓ Face registration completed successfully!")
    else:
        print("\n✗ Face registration failed. Please check the error messages above.")
    
    input("\nPress Enter to continue...")


def recognize_from_image():
    """Recognize faces from an image file."""
    print("\n" + "="*60)
    print("FACE RECOGNITION FROM IMAGE")
    print("="*60)
    
    # Get image path from user
    image_path = input("\nEnter the path to the image file: ").strip()
    
    if not image_path:
        print("Error: No image path provided")
        input("\nPress Enter to continue...")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        input("\nPress Enter to continue...")
        return
    
    # Create recognizer and load database
    recognizer = FaceRecognizer()
    
    if not recognizer.load_database():
        print("\nError: Could not load face database.")
        print("Please register faces first (Option 1)")
        input("\nPress Enter to continue...")
        return
    
    # Recognize faces
    recognizer.recognize_from_image_file(image_path)
    
    input("\nPress Enter to continue...")


def webcam_recognition():
    """Real-time face recognition from webcam."""
    print("\n" + "="*60)
    print("REAL-TIME WEBCAM RECOGNITION")
    print("="*60)
    
    # Create recognizer and load database
    recognizer = FaceRecognizer()
    
    if not recognizer.load_database():
        print("\nError: Could not load face database.")
        print("Please register faces first (Option 1)")
        input("\nPress Enter to continue...")
        return
    
    # Start webcam recognition
    recognizer.recognize_from_webcam()
    
    input("\nPress Enter to continue...")


def list_registered_faces():
    """List all registered faces."""
    print("\n" + "="*60)
    print("REGISTERED FACES")
    print("="*60)
    
    encoder = FaceEncoder()
    names = encoder.get_registered_names()
    
    if not names:
        print("\nNo faces registered yet.")
        print("Please register faces first (Option 1)")
    else:
        print(f"\nTotal unique persons registered: {len(names)}")
        print("\nRegistered names:")
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
    
    input("\nPress Enter to continue...")


def main():
    """Main application loop."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                register_faces()
            elif choice == "2":
                recognize_from_image()
            elif choice == "3":
                webcam_recognition()
            elif choice == "4":
                list_registered_faces()
            elif choice == "5":
                print("\nThank you for using Face Recognition System!")
                print("Goodbye!\n")
                sys.exit(0)
            else:
                print("\nInvalid choice. Please enter a number between 1 and 5.")
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
