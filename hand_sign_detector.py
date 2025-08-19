import cv2
import numpy as np
import json
import os
import time
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class HandSignDetector:
    def __init__(self):
        """Initialize the hand sign detector for static poses"""
        
        # Sign database
        self.signs_file = "hand_signs.json"
        self.trained_signs = self.load_signs()
        
        # Recognition parameters
        self.similarity_threshold = 0.85  # Higher threshold for static poses
        self.last_recognized_sign = ""
        self.last_recognition_time = 0
        self.recognition_cooldown = 0.0  # No cooldown - immediate recognition
        
        # Smoothing for stable recognition
        self.recent_predictions = []
        self.prediction_history_size = 5
        
        # Current sign being recorded
        self.is_recording = False
        self.current_sign_name = ""
        self.recorded_poses = []
        
        # Emergency sequence detection (palm → help → fist)
        self.sequence_detection_enabled = True
        self.target_sequence = ["palm", "help", "fist"]
        self.current_sequence = []
        self.sequence_timeout = 10.0  # seconds to complete sequence
        self.sequence_start_time = 0
        self.last_sequence_trigger = 0
        self.sequence_cooldown = 30.0  # 30 seconds between email alerts
        
        # Email configuration
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",  # Will be set by user
            "sender_password": "",  # Will be set by user
            "recipient_email": ""  # Will be set by user
        }
        
        # Load saved email configuration
        self.load_email_config()
        
    def load_signs(self):
        """Load trained signs from file"""
        if os.path.exists(self.signs_file):
            try:
                with open(self.signs_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def load_email_config(self):
        """Load email configuration from file"""
        email_config_file = "email_config.json"
        if os.path.exists(email_config_file):
            try:
                with open(email_config_file, 'r') as f:
                    config_data = json.load(f)
                
                self.email_config["sender_email"] = config_data.get("sender_email", "")
                self.email_config["sender_password"] = config_data.get("sender_password", "")
                self.email_config["recipient_email"] = config_data.get("recipient_email", "")
                self.email_config["smtp_server"] = config_data.get("smtp_server", "smtp.gmail.com")
                self.email_config["smtp_port"] = config_data.get("smtp_port", 587)
                
                if all([self.email_config["sender_email"], 
                       self.email_config["sender_password"], 
                       self.email_config["recipient_email"]]):
                    print(f"📧 Email config loaded: {self.email_config['sender_email']} → {self.email_config['recipient_email']}")
            except Exception as e:
                print(f"⚠️ Could not load email config: {e}")
    
    def save_signs(self):
        """Save trained signs to file"""
        with open(self.signs_file, 'w') as f:
            json.dump(self.trained_signs, f, indent=2)
    
    def extract_hand_features(self, landmarks):
        """
        Extract features from hand landmarks for sign recognition
        
        Args:
            landmarks: List of hand landmarks from MediaPipe
            
        Returns:
            features: Dictionary of hand features
        """
        if not landmarks or len(landmarks) < 21:
            return None
        
        features = {}
        
        # Finger tip and joint positions
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        pip_ids = [3, 6, 10, 14, 18]  # PIP joints
        mcp_ids = [2, 5, 9, 13, 17]   # MCP joints
        
        # 1. Finger states (up/down)
        finger_states = []
        
        # Thumb (special case - compare x coordinates)
        thumb_tip = landmarks[tip_ids[0]]
        thumb_joint = landmarks[tip_ids[0] - 1]
        finger_states.append(1 if thumb_tip[1] > thumb_joint[1] else 0)
        
        # Other fingers (compare y coordinates)
        for i in range(1, 5):
            tip = landmarks[tip_ids[i]]
            pip = landmarks[pip_ids[i]]
            finger_states.append(1 if tip[2] < pip[2] else 0)
        
        features['finger_states'] = finger_states
        
        # 2. Finger angles and relationships
        wrist = landmarks[0]
        
        # Distances from wrist to fingertips (normalized)
        tip_distances = []
        for tip_id in tip_ids:
            tip = landmarks[tip_id]
            distance = np.sqrt((tip[1] - wrist[1])**2 + (tip[2] - wrist[2])**2)
            tip_distances.append(distance)
        
        # Normalize distances
        if max(tip_distances) > 0:
            tip_distances = [d / max(tip_distances) for d in tip_distances]
        features['tip_distances'] = tip_distances
        
        # 3. Finger spread (angles between adjacent fingers)
        finger_spreads = []
        for i in range(4):  # Between adjacent fingers
            tip1 = landmarks[tip_ids[i]]
            tip2 = landmarks[tip_ids[i + 1]]
            
            # Calculate angle
            dx = tip2[1] - tip1[1]
            dy = tip2[2] - tip1[2]
            angle = np.arctan2(dy, dx)
            finger_spreads.append(angle)
        
        features['finger_spreads'] = finger_spreads
        
        # 4. Hand orientation
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Hand direction vector
        hand_dx = middle_tip[1] - index_tip[1]
        hand_dy = middle_tip[2] - index_tip[2]
        hand_angle = np.arctan2(hand_dy, hand_dx)
        features['hand_angle'] = hand_angle
        
        # 5. Finger curl levels (how bent each finger is)
        curl_levels = []
        for i in range(5):
            if i == 0:  # Thumb
                mcp = landmarks[mcp_ids[i]]
                pip = landmarks[pip_ids[i]]
                tip = landmarks[tip_ids[i]]
            else:  # Other fingers
                mcp = landmarks[mcp_ids[i]]
                pip = landmarks[pip_ids[i]]
                tip = landmarks[tip_ids[i]]
            
            # Calculate curl based on relative positions
            curl = self.calculate_finger_curl(mcp, pip, tip)
            curl_levels.append(curl)
        
        features['curl_levels'] = curl_levels
        
        return features
    
    def calculate_finger_curl(self, mcp, pip, tip):
        """Calculate how much a finger is curled (0 = straight, 1 = fully curled)"""
        # Vector from MCP to PIP
        vec1 = [pip[1] - mcp[1], pip[2] - mcp[2]]
        # Vector from PIP to TIP
        vec2 = [tip[1] - pip[1], tip[2] - pip[2]]
        
        # Calculate angle between vectors
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Convert to curl level (0 = straight, 1 = fully curled)
        curl = 1 - (angle / np.pi)
        return max(0, min(1, curl))
    
    def start_recording_sign(self, sign_name):
        """Start recording a new sign"""
        self.is_recording = True
        self.current_sign_name = sign_name
        self.recorded_poses = []
        print(f"🔴 Recording sign: {sign_name}")
    
    def add_pose_sample(self, landmarks):
        """Add a pose sample while recording"""
        if self.is_recording and landmarks:
            features = self.extract_hand_features(landmarks)
            if features:
                self.recorded_poses.append(features)
                print(f"📸 Captured pose sample {len(self.recorded_poses)}")
    
    def stop_recording_sign(self):
        """Stop recording and save the sign"""
        if self.is_recording and len(self.recorded_poses) >= 3:
            # Average the recorded poses for stability
            averaged_features = self.average_poses(self.recorded_poses)
            
            # Check if sign already exists
            if self.current_sign_name in self.trained_signs:
                # Add to existing examples
                if not isinstance(self.trained_signs[self.current_sign_name], list):
                    # Convert single example to list
                    old_sign = self.trained_signs[self.current_sign_name]
                    self.trained_signs[self.current_sign_name] = [old_sign]
                
                self.trained_signs[self.current_sign_name].append(averaged_features)
                example_count = len(self.trained_signs[self.current_sign_name])
                print(f"✅ Added example #{example_count} for '{self.current_sign_name}'")
            else:
                # New sign
                self.trained_signs[self.current_sign_name] = averaged_features
                print(f"✅ New sign '{self.current_sign_name}' saved")
            
            self.save_signs()
            
        elif self.is_recording:
            print(f"❌ Need at least 3 pose samples. Got {len(self.recorded_poses)}")
        
        self.is_recording = False
        self.current_sign_name = ""
        self.recorded_poses = []
    
    def average_poses(self, poses):
        """Average multiple pose samples for stability"""
        if not poses:
            return None
        
        # Initialize averaged features
        avg_features = {}
        
        # Average each feature type
        for key in poses[0].keys():
            if key == 'finger_states':
                # For binary states, use majority vote
                states = [pose[key] for pose in poses]
                avg_states = []
                for i in range(len(states[0])):
                    finger_votes = [state[i] for state in states]
                    avg_states.append(1 if sum(finger_votes) > len(finger_votes) / 2 else 0)
                avg_features[key] = avg_states
            else:
                # For continuous values, use mean
                values = [pose[key] for pose in poses]
                if isinstance(values[0], list):
                    # List of values
                    avg_values = []
                    for i in range(len(values[0])):
                        avg_values.append(np.mean([v[i] for v in values]))
                    avg_features[key] = avg_values
                else:
                    # Single value
                    avg_features[key] = np.mean(values)
        
        return avg_features
    
    def recognize_sign(self, landmarks):
        """
        Recognize a hand sign from current landmarks
        
        Args:
            landmarks: Current hand landmarks
            
        Returns:
            sign_name: Name of recognized sign or None
        """
        if not landmarks or not self.trained_signs:
            return None
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return None
        
        # Extract features from current pose
        current_features = self.extract_hand_features(landmarks)
        if not current_features:
            return None
        
        # Compare with trained signs
        best_match = None
        best_similarity = 0
        
        for sign_name, trained_data in self.trained_signs.items():
            # Handle both single example and multiple examples
            if isinstance(trained_data, dict):
                # Single example
                similarities = [self.calculate_sign_similarity(current_features, trained_data)]
            else:
                # Multiple examples
                similarities = [self.calculate_sign_similarity(current_features, example) 
                              for example in trained_data]
            
            # Use best similarity
            sign_similarity = max(similarities)
            
            if sign_similarity > best_similarity and sign_similarity > self.similarity_threshold:
                best_similarity = sign_similarity
                best_match = sign_name
        
        # Add to recent predictions for smoothing
        self.recent_predictions.append(best_match if best_match else "none")
        if len(self.recent_predictions) > self.prediction_history_size:
            self.recent_predictions.pop(0)
        
        # Use most common recent prediction
        if len(self.recent_predictions) >= 3:
            prediction_counts = Counter(self.recent_predictions)
            most_common = prediction_counts.most_common(1)[0]
            if most_common[0] != "none" and most_common[1] >= 2:  # At least 2 out of recent predictions
                if most_common[0] != self.last_recognized_sign:  # New recognition
                    self.last_recognized_sign = most_common[0]
                    self.last_recognition_time = current_time
                    print(f"🎯 Recognized sign: {most_common[0]} (similarity: {best_similarity:.2f})")
                    
                    # Update emergency sequence detection
                    self.update_sequence(most_common[0])
                    
                    return most_common[0]
        
        return None
    
    def calculate_sign_similarity(self, features1, features2):
        """Calculate similarity between two hand sign features"""
        if not features1 or not features2:
            return 0
        
        total_similarity = 0
        weight_sum = 0
        
        # 1. Finger states similarity (high weight)
        if 'finger_states' in features1 and 'finger_states' in features2:
            states1 = features1['finger_states']
            states2 = features2['finger_states']
            matches = sum(1 for a, b in zip(states1, states2) if a == b)
            finger_similarity = matches / len(states1)
            total_similarity += finger_similarity * 3.0  # High weight
            weight_sum += 3.0
        
        # 2. Curl levels similarity (medium weight)
        if 'curl_levels' in features1 and 'curl_levels' in features2:
            curl1 = np.array(features1['curl_levels'])
            curl2 = np.array(features2['curl_levels'])
            curl_diff = np.mean(np.abs(curl1 - curl2))
            curl_similarity = 1 - curl_diff
            total_similarity += curl_similarity * 2.0  # Medium weight
            weight_sum += 2.0
        
        # 3. Tip distances similarity (medium weight)
        if 'tip_distances' in features1 and 'tip_distances' in features2:
            dist1 = np.array(features1['tip_distances'])
            dist2 = np.array(features2['tip_distances'])
            dist_diff = np.mean(np.abs(dist1 - dist2))
            dist_similarity = 1 - dist_diff
            total_similarity += dist_similarity * 2.0  # Medium weight
            weight_sum += 2.0
        
        # 4. Finger spreads similarity (low weight)
        if 'finger_spreads' in features1 and 'finger_spreads' in features2:
            spread1 = np.array(features1['finger_spreads'])
            spread2 = np.array(features2['finger_spreads'])
            spread_diff = np.mean(np.abs(spread1 - spread2))
            spread_similarity = 1 - (spread_diff / np.pi)  # Normalize by pi
            total_similarity += spread_similarity * 1.0  # Low weight
            weight_sum += 1.0
        
        return total_similarity / weight_sum if weight_sum > 0 else 0
    
    def get_predefined_signs(self):
        """Get a list of common hand signs to train"""
        return [
            "thumbs_up",
            "thumbs_down", 
            "peace_sign",
            "ok_sign",
            "fist",
            "open_hand",
            "pointing",
            "rock_sign",
            "call_me",
            "i_love_you"
        ]
    
    def draw_sign_info(self, img):
        """Draw sign recognition information on image"""
        # Show recent recognition
        if self.last_recognized_sign:
            time_since = time.time() - self.last_recognition_time
            if time_since < 3.0:  # Show for 3 seconds
                y_offset = img.shape[0] - 100
                cv2.rectangle(img, (10, y_offset), (300, y_offset + 50), (0, 255, 0), -1)
                cv2.putText(img, f"Sign: {self.last_recognized_sign}", 
                           (15, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return img
    
    def draw_recording_status(self, img):
        """Draw recording status on image"""
        if self.is_recording:
            # Recording indicator
            cv2.rectangle(img, (10, 10), (300, 50), (0, 0, 255), -1)
            cv2.putText(img, f"Recording: {self.current_sign_name}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def configure_email(self, sender_email, sender_password, recipient_email):
        """Configure email settings for emergency alerts"""
        self.email_config["sender_email"] = sender_email
        self.email_config["sender_password"] = sender_password
        self.email_config["recipient_email"] = recipient_email
        print(f"✅ Email configured: {sender_email} → {recipient_email}")
        
        # Save email config to file for persistence
        email_config_file = "email_config.json"
        config_data = {
            "sender_email": sender_email,
            "sender_password": sender_password,
            "recipient_email": recipient_email,
            "smtp_server": self.email_config["smtp_server"],
            "smtp_port": self.email_config["smtp_port"]
        }
        
        try:
            with open(email_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"💾 Email configuration saved to {email_config_file}")
        except Exception as e:
            print(f"⚠️ Could not save email config: {e}")
    
    def send_emergency_email(self):
        """Send emergency email alert"""
        if not all([self.email_config["sender_email"], 
                   self.email_config["sender_password"], 
                   self.email_config["recipient_email"]]):
            print("❌ Email not configured! Use configure_email() first")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config["sender_email"]
            msg['To'] = self.email_config["recipient_email"]
            msg['Subject'] = "🚨 EMERGENCY ALERT - Hand Sign Sequence Detected"
            
            body = """
🚨 EMERGENCY ALERT 🚨

The hand sign emergency sequence (Palm → Help → Fist) has been detected!

This automated alert was triggered by the hand sign recognition system.
Time: {}

Please check on the person immediately.

---
Hand Sign Recognition System
            """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["sender_email"], self.email_config["sender_password"])
            text = msg.as_string()
            server.sendmail(self.email_config["sender_email"], self.email_config["recipient_email"], text)
            server.quit()
            
            print("📧 Emergency email sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
    
    def update_sequence(self, recognized_sign):
        """Update the emergency sequence detection"""
        if not self.sequence_detection_enabled or not recognized_sign:
            return
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_sequence_trigger < self.sequence_cooldown:
            return
        
        # Start new sequence if empty or timeout
        if not self.current_sequence or (current_time - self.sequence_start_time > self.sequence_timeout):
            if recognized_sign == self.target_sequence[0]:  # First sign (palm)
                self.current_sequence = [recognized_sign]
                self.sequence_start_time = current_time
                print(f"🔄 Sequence started: {recognized_sign}")
            else:
                self.current_sequence = []
            return
        
        # Continue sequence
        expected_next = self.target_sequence[len(self.current_sequence)]
        if recognized_sign == expected_next:
            self.current_sequence.append(recognized_sign)
            print(f"🔄 Sequence progress: {' → '.join(self.current_sequence)}")
            
            # Check if sequence complete
            if len(self.current_sequence) == len(self.target_sequence):
                print("🚨 EMERGENCY SEQUENCE DETECTED!")
                self.send_emergency_email()
                self.last_sequence_trigger = current_time
                self.current_sequence = []
        else:
            # Wrong sign, reset sequence
            if recognized_sign == self.target_sequence[0]:  # Start over with first sign
                self.current_sequence = [recognized_sign]
                self.sequence_start_time = current_time
                print(f"🔄 Sequence restarted: {recognized_sign}")
            else:
                self.current_sequence = []
    
    def draw_sequence_status(self, img):
        """Draw sequence detection status on image"""
        if not self.sequence_detection_enabled:
            return img
        
        # Show emergency alert if recently triggered
        current_time = time.time()
        if current_time - self.last_sequence_trigger < 5.0:  # Show for 5 seconds
            # Create flashing red background
            flash_intensity = int(255 * (1 - (current_time - self.last_sequence_trigger) / 5.0))
            flash_color = (0, 0, flash_intensity)
            
            # Draw emergency alert box
            h, w = img.shape[:2]
            cv2.rectangle(img, (w//2 - 200, h//2 - 100), (w//2 + 200, h//2 + 100), flash_color, -1)
            cv2.rectangle(img, (w//2 - 200, h//2 - 100), (w//2 + 200, h//2 + 100), (255, 255, 255), 3)
            
            # Draw emergency text
            cv2.putText(img, "🚨 EMERGENCY ALERT 🚨", 
                       (w//2 - 180, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Sequence Detected!", 
                       (w//2 - 150, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Email Sent!", 
                       (w//2 - 100, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Time: {time.strftime('%H:%M:%S')}", 
                       (w//2 - 80, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img