#!/usr/bin/env python3
"""
Email Configuration Script for Hand Sign Emergency Alerts
"""

from hand_sign_detector import HandSignDetector

def main():
    print("📧 EMERGENCY EMAIL ALERT CONFIGURATION")
    print("=" * 60)
    print("This script configures email alerts for the emergency hand sign sequence:")
    print("🚨 Palm → Help → Fist")
    print("")
    print("When this sequence is detected, an emergency email will be sent.")
    print("=" * 60)
    
    # Initialize detector
    detector = HandSignDetector()
    
    try:
        print("\n📝 Enter your email configuration:")
        print("(Leave blank to cancel)")
        print("")
        
        sender_email = input("Sender email (Gmail recommended): ").strip()
        if not sender_email:
            print("❌ Configuration cancelled")
            return
        
        print("\n⚠️  IMPORTANT: For Gmail, you need an App Password!")
        print("   How to get a Gmail App Password:")
        print("   1. Go to your Google Account settings")
        print("   2. Security → 2-Step Verification (enable if not already)")
        print("   3. Security → App passwords")
        print("   4. Generate a new app password for 'Hand Sign Detection'")
        print("   5. Use that 16-character password below")
        print("")
        
        sender_password = input("App Password (not your regular password): ").strip()
        if not sender_password:
            print("❌ Configuration cancelled")
            return
        
        recipient_email = input("Emergency contact email: ").strip()
        if not recipient_email:
            print("❌ Configuration cancelled")
            return
        
        # Configure
        detector.configure_email(sender_email, sender_password, recipient_email)
        
        # Test
        print("\n🧪 Testing email configuration...")
        test = input("Send test email now? (y/n): ").strip().lower()
        
        if test == 'y':
            print("Sending test email...")
            # Temporarily override the subject for test
            original_method = detector.send_emergency_email
            
            def send_test_email():
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                import time
                
                msg = MIMEMultipart()
                msg['From'] = detector.email_config["sender_email"]
                msg['To'] = detector.email_config["recipient_email"]
                msg['Subject'] = "✅ Test Email - Hand Sign Alert System"
                
                body = f"""
✅ TEST EMAIL SUCCESS!

Your hand sign emergency alert system is configured correctly.

Configuration:
• Sender: {detector.email_config["sender_email"]}
• Recipient: {detector.email_config["recipient_email"]}
• Emergency Sequence: Palm → Help → Fist
• Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

The system is now ready to send emergency alerts when the hand sign sequence is detected.

---
Hand Sign Recognition System
Test Email
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                try:
                    server = smtplib.SMTP(detector.email_config["smtp_server"], detector.email_config["smtp_port"])
                    server.starttls()
                    server.login(detector.email_config["sender_email"], detector.email_config["sender_password"])
                    text = msg.as_string()
                    server.sendmail(detector.email_config["sender_email"], detector.email_config["recipient_email"], text)
                    server.quit()
                    return True
                except Exception as e:
                    print(f"❌ Failed to send test email: {e}")
                    return False
            
            if send_test_email():
                print("✅ Test email sent successfully!")
                print(f"📧 Check {recipient_email} for the test message")
            else:
                print("❌ Test email failed. Please check your configuration.")
                return
        
        print("\n" + "=" * 60)
        print("✅ EMAIL ALERT SYSTEM CONFIGURED!")
        print("=" * 60)
        print("🚨 Emergency sequence: Palm → Help → Fist")
        print(f"📧 Alerts will be sent to: {recipient_email}")
        print("🔧 Configuration saved in the detector")
        print("")
        print("Now you can run:")
        print("• python sign_trainer.py (for training and recognition)")
        print("• python finger_counter_with_signs.py (for finger counting + signs)")
        print("")
        print("The emergency alert system is active in both applications!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Configuration cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()