import cv2
import time
from hand_detector import HandDetector
from hand_sign_detector import HandSignDetector

class FingerCounterWithSigns:
    def __init__(self):
        """Initialize the enhanced finger counter with sign recognition"""
        self.hand_detector = HandDetector(detection_confidence=0.75)
        self.sign_detector = HandSignDetector()
        
        # Try different camera indices for DroidCam/iPhone webcam
        self.cap = None
        for i in range(3):  # Try cameras 0, 1, 2
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                # Set properties before reading frame
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                test_cap.set(cv2.CAP_PROP_FPS, 30)
                
                ret, frame = test_cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.cap = test_cap
                    print(f"✅ Using camera {i}")
                    break
                test_cap.release()
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # Fallback
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("❌ Error: Could not open camera")
        
        print("✅ Camera opened successfully")
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera resolution: {width}x{height}")
        
        # For FPS calculation
        self.prev_time = 0
        
        # Finger count history for smoothing
        self.finger_count_history = []
        self.history_size = 5
        
        # Display modes
        self.show_landmarks = True
        
    def get_smoothed_count(self, current_count):
        """Smooth the finger count to reduce flickering"""
        self.finger_count_history.append(current_count)
        
        if len(self.finger_count_history) > self.history_size:
            self.finger_count_history.pop(0)
        
        # Return the most common count in recent history
        if self.finger_count_history:
            return max(set(self.finger_count_history), key=self.finger_count_history.count)
        return current_count
    
    def draw_enhanced_ui(self, img, finger_count, fps):
        """Draw enhanced UI with finger count and sign recognition"""
        # Finger count display
        cv2.rectangle(img, (50, 50), (300, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Fingers: {finger_count}', (60, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        return img
    
    def draw_number_visualization(self, img, finger_count):
        """Draw visual representation of finger count"""
        start_x = img.shape[1] - 300
        start_y = 80
        circle_radius = 15
        circle_spacing = 40
        
        # Background
        cv2.rectangle(img, (start_x - 40, start_y - 40), 
                     (start_x + 200, start_y + 80), (50, 50, 50), cv2.FILLED)
        
        # Title
        cv2.putText(img, 'Count:', (start_x - 30, start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw circles for each finger (0-5)
        for i in range(6):
            x = start_x + (i % 3) * circle_spacing
            y = start_y + (i // 3) * circle_spacing
            
            if i < finger_count:
                cv2.circle(img, (x, y), circle_radius, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(i+1), (x-5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            else:
                cv2.circle(img, (x, y), circle_radius, (100, 100, 100), 2)
                cv2.putText(img, str(i), (x-5, y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Run the enhanced finger counter with sign recognition"""
        print("🚀 Enhanced Finger Counter with Hand Sign Recognition")
        print("=" * 70)
        print("Features:")
        print("• Real-time finger counting")
        print("• Static hand sign recognition (thumbs up, peace, OK, etc.)")
        print("• High-quality iPhone camera support via DroidCam")
        print("• Interactive sign training mode")
        print("=" * 70)
        
        # Create window
        window_name = "Finger Counter + Hand Signs"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print(f"✅ Created window: {window_name}")
        
        # Initialize window
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Error: Could not read first frame")
            return
        
        first_frame = cv2.flip(first_frame, 1)
        cv2.putText(first_frame, 'Loading...', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, first_frame)
        cv2.waitKey(100)
        print("✅ Window should now be visible")
        
        while True:
            ret, img = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            
            # Detect hands
            img = self.hand_detector.find_hands(img, draw=self.show_landmarks)
            landmarks = self.hand_detector.find_position(img, draw=False)
            
            # Count fingers
            finger_count = 0
            if len(landmarks) != 0:
                finger_count = self.hand_detector.count_fingers()
                finger_count = self.get_smoothed_count(finger_count)
            
            # Try to recognize hand signs
            if landmarks:
                recognized_sign = self.sign_detector.recognize_sign(landmarks)
            
            # Calculate FPS (not displayed)
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = curr_time
            
            # Draw UI
            img = self.draw_enhanced_ui(img, finger_count, fps)
            
            # Draw sign recognition info
            img = self.sign_detector.draw_sign_info(img)
            img = self.sign_detector.draw_sequence_status(img)
            
            # Display
            cv2.imshow(window_name, img)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('l'):  # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('s'):  # Open sign trainer
                print("Opening sign trainer...")
                break
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Open sign trainer if requested
        if key == ord('s'):
            try:
                from sign_trainer import SignTrainer
                trainer = SignTrainer()
                trainer.run()
            except Exception as e:
                print(f"Error opening sign trainer: {e}")
        
        print("Application closed!")

if __name__ == "__main__":
    try:
        app = FingerCounterWithSigns()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible")