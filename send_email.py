import smtplib
import ssl
from email.message import EmailMessage
import os
import sys

def send_email(to_email, subject, body):
    # This is a stub for the autonomous pipeline to use.
    # In a real environment, we'd use environment variables for SMTP credentials.
    # For now, I'll log the intent to the laboratory console.
    print(f"--- EMAIL TRANSMISSION INITIATED ---")
    print(f"To: {to_email}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print(f"------------------------------------")
    
    # Check for credentials in environment
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not smtp_user or not smtp_pass:
        print("Error: SMTP credentials missing (SMTP_USER/SMTP_PASS).")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 send_email.py <to> <subject> <body>")
        sys.exit(1)
    send_email(sys.argv[1], sys.argv[2], sys.argv[3])
