import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe
        
        Args:
            mode: Static image mode or video mode
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tip and pip landmark IDs for each finger
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.pip_ids = [3, 6, 10, 14, 18]  # Thumb, Index, Middle, Ring, Pinky PIPs
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image and optionally draw landmarks
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks
            
        Returns:
            img: Image with or without hand landmarks drawn
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find the position of hand landmarks
        
        Args:
            img: Input image
            hand_no: Hand number (0 for first hand, 1 for second)
            draw: Whether to draw landmark positions
            
        Returns:
            lm_list: List of landmark positions [id, x, y]
        """
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return self.lm_list
    
    def fingers_up(self):
        """
        Determine which fingers are up
        
        Returns:
            fingers: List of 1s and 0s representing which fingers are up
        """
        fingers = []
        
        if len(self.lm_list) != 0:
            # Thumb - special case (compare x coordinates)
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other four fingers (compare y coordinates)
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.pip_ids[id]][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    def count_fingers(self):
        """
        Count the number of fingers that are up
        
        Returns:
            total_fingers: Total number of fingers up
        """
        fingers = self.fingers_up()
        total_fingers = fingers.count(1)
        return total_fingers