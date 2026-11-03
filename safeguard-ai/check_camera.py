"""Checks a camera link and says in plain words what is wrong.

    python check_camera.py                                  uses HOME_CAM_URL from .env
    python check_camera.py http://192.168.31.105:8080/video checks that link
"""
import os
import re
import socket
import sys
import time
from urllib.parse import urlsplit

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2


def mask(u):
    return re.sub(r"//[^@/]+@", "//***@", u)


def my_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


url = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("HOME_CAM_URL", "")).strip()
if not url or "${" in url:
    print("No camera link found. Put HOME_CAM_URL=... in the .env file (spelled exactly like that),")
    print("or run:  python check_camera.py http://YOUR-PHONE-ADDRESS:8080/video")
    sys.exit(1)

print("Checking:", mask(url), "\n")
parts = urlsplit(url)
ports = {"http": 80, "https": 443, "rtsp": 554, "rtmp": 1935}
if parts.scheme in ports and parts.hostname:
    host, port = parts.hostname, parts.port or ports[parts.scheme]
    mine = my_address()
    print(f"Step 1. This laptop's network address: {mine}")
    print(f"        The camera's address:          {host}")
    if mine and re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        if mine.split(".")[:3] != host.split(".")[:3]:
            print("        WARNING: the first three numbers differ. The laptop and the camera are probably on")
            print("        DIFFERENT networks (different Wi-Fi, a guest network, a VPN or a hotspot).")
        else:
            print("        Same network. Good.")
    print(f"\nStep 2. Can the laptop reach {host} on port {port}?")
    try:
        socket.create_connection((host, port), timeout=6).close()
        print("        Yes, something answered.")
    except (socket.timeout, TimeoutError):
        print("        NO ANSWER (timed out). The address is wrong or the phone is on a different network.")
        print("        - Check the address shown on the phone under 'Start server'. It may have changed.")
        print("        - Make sure the phone is on the same Wi-Fi as the laptop, with mobile data off.")
        print("        - Try the phone's hotspot: connect the laptop to it, then use the new address.")
        sys.exit(2)
    except ConnectionRefusedError:
        print("        The phone answered, but nothing is listening on that port.")
        print("        - Open IP Webcam on the phone and tap 'Start server' (keep the screen on).")
        print("        - Check the port number in the link matches the one the app shows (usually 8080).")
        sys.exit(2)
    except OSError as e:
        print(f"        Network error: {e}")
        print("        Check the Wi-Fi connection on both devices.")
        sys.exit(2)
else:
    print("(Not a network link, skipping the network checks.)")

print("\nStep 3. Trying to read a picture ...")
cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
if not cap.isOpened():
    print("        Could not open the video. The address answers but the video path is wrong.")
    print("        For IP Webcam the link must end in /video. Another one to try: /videofeed")
    sys.exit(3)
frame, end = None, time.time() + 10
while frame is None and time.time() < end:
    ok, f = cap.read()
    frame = f if ok else None
cap.release()
if frame is None:
    print("        Connected, but no picture arrived within 10 seconds. Check that IP Webcam is streaming.")
    sys.exit(3)
print(f"        SUCCESS. A picture arrived ({frame.shape[1]}x{frame.shape[0]}). The link works.")
print("        If the program still shows errors, close it and start start_safeguard.bat again.")
