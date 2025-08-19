import cv2
import time
from hand_detector import HandDetector
from hand_sign_detector import HandSignDetector

class SignTrainer:
    def __init__(self):
        """Initialize the sign training application"""
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
        
        # Check camera
        if not self.cap.isOpened():
            raise RuntimeError("❌ Error: Could not open camera")
        
        # Training state
        self.training_mode = True
        self.current_sign_index = 0
        self.predefined_signs = self.sign_detector.get_predefined_signs()
        
        # UI state
        self.prev_time = 0
        
        # MANUAL EMAIL CONFIGURATION - Add your email details here
        self.sign_detector.configure_email(
            sender_email="adamsirri45@gmail.com",  # Your Gmail
            sender_password="hwfnrrwanymnlsju",  # Your Gmail App Password
            recipient_email="antunietta.durazzo@sciencespo.fr"  # Emergency contact
        )
        
    def draw_training_ui(self, img):
        """Draw training interface"""
        return img
    
    def draw_sign_list(self, img):
        """Draw list of trained signs with example counts"""
        return img
    
    def handle_training_input(self, key):
        """Handle keyboard input during training"""
        if key == ord(' '):  # Space - start recording
            if not self.sign_detector.is_recording:
                current_sign = self.predefined_signs[self.current_sign_index]
                self.sign_detector.start_recording_sign(current_sign)
        
        elif key == ord('\r') or key == ord('\n'):  # Enter - save recording
            if self.sign_detector.is_recording:
                self.sign_detector.stop_recording_sign()
        
        elif key == ord('n'):  # Next sign
            if not self.sign_detector.is_recording:
                self.current_sign_index = (self.current_sign_index + 1) % len(self.predefined_signs)
                print(f"Switched to sign: {self.predefined_signs[self.current_sign_index]}")
        
        elif key == ord('c'):  # Custom sign
            if not self.sign_detector.is_recording:
                sign_name = input("Enter custom sign name: ")
                if sign_name:
                    self.sign_detector.start_recording_sign(sign_name)
        
        elif key == ord('r'):  # Recognition mode
            self.training_mode = False
            print("Switched to recognition mode")
    
    def configure_email_alert(self):
        """Configure email settings for emergency alerts"""
        print("\n" + "="*50)
        print("📧 EMAIL ALERT CONFIGURATION")
        print("="*50)
        print("Configure emergency email alerts for Palm → Help → Fist sequence")
        print("")
        
        try:
            sender_email = input("Enter sender email (Gmail): ").strip()
            if not sender_email:
                print("❌ Email configuration cancelled")
                return
            
            # For Gmail, you need an App Password, not your regular password
            print("\n⚠️  IMPORTANT: For Gmail, use an App Password!")
            print("   1. Go to Google Account settings")
            print("   2. Enable 2-Factor Authentication")
            print("   3. Generate an App Password for this application")
            print("   4. Use the App Password below (not your regular password)")
            print("")
            
            sender_password = input("Enter App Password: ").strip()
            if not sender_password:
                print("❌ Email configuration cancelled")
                return
            
            recipient_email = input("Enter recipient email (emergency contact): ").strip()
            if not recipient_email:
                print("❌ Email configuration cancelled")
                return
            
            # Configure the detector
            self.sign_detector.configure_email(sender_email, sender_password, recipient_email)
            
            # Test email (optional)
            test = input("\nSend test email? (y/n): ").strip().lower()
            if test == 'y':
                print("Sending test email...")
                if self.sign_detector.send_emergency_email():
                    print("✅ Test email sent successfully!")
                else:
                    print("❌ Test email failed. Check your settings.")
            
            print("="*50)
            print("📧 Email alert system is now active!")
            print("🚨 Emergency sequence: Palm → Help → Fist")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n❌ Email configuration cancelled")
        except Exception as e:
            print(f"❌ Error configuring email: {e}")
    
    def run(self):
        """Run the sign training application"""
        print("🖐️ Hand Sign Training Application")
        print("=" * 60)
        print("NEW: Static Hand Sign Recognition!")
        print("• Train hand signs like thumbs up, peace sign, OK sign, etc.")
        print("• System recognizes STATIC poses, not movements")
        print("• Hold poses steady for best recognition")
        print("")
        print("INSTRUCTIONS:")
        print("1. Training Mode: Press SPACE to start, hold pose, press ENTER to save")
        print("2. Train multiple examples of each sign for better accuracy")
        print("3. Recognition Mode: Make signs and hold them steady")
        print("4. Press 'e' to configure emergency email alerts")
        print("5. Press 'q' to quit")
        print("")
        print("🚨 EMERGENCY FEATURE: Palm → Help → Fist sequence sends email alert!")
        print("=" * 60)
        
        # Create window
        window_name = "Hand Sign Trainer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while True:
            ret, img = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            
            # Detect hands
            img = self.hand_detector.find_hands(img, draw=True)
            landmarks = self.hand_detector.find_position(img, draw=False)
            
            # Handle recording
            if self.sign_detector.is_recording and landmarks:
                self.sign_detector.add_pose_sample(landmarks)
            
            # Try to recognize signs (only in recognition mode)
            if not self.training_mode and not self.sign_detector.is_recording and landmarks:
                recognized = self.sign_detector.recognize_sign(landmarks)
            
            # Draw UI elements
            img = self.draw_training_ui(img)
            img = self.draw_sign_list(img)
            img = self.sign_detector.draw_recording_status(img)
            img = self.sign_detector.draw_sign_info(img)
            img = self.sign_detector.draw_sequence_status(img)
            
            # Calculate FPS (not displayed)
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
            self.prev_time = curr_time
            
            # Display
            cv2.imshow(window_name, img)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t') and not self.training_mode:
                self.training_mode = True
                print("Switched to training mode")
            elif key == ord('e') and not self.training_mode:
                self.configure_email_alert()
            elif self.training_mode:
                self.handle_training_input(key)
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sign training application closed!")

if __name__ == "__main__":
    try:
        trainer = SignTrainer()
        trainer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible")