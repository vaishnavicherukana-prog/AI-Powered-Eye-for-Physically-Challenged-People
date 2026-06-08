import time
import pytesseract
import os
import cv2
from threading import Thread
from PIL import Image

# Optional: Set path to tesseract executable (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to convert text to audio using espeak
def txt_audio(msg):
    print('audio begin')
    s = f'espeak "{msg}"'
    os.system(s)

# Function to convert image to text and speak it
def img_txt(image):
    content = pytesseract.image_to_string(image)
    print('audio begin')
    s = f'espeak "{content}"'
    os.system(s)
    print(content)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Showing camera preview for 5 seconds...")

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Camera Preview - Press Q to quit', frame)

    # Exit after 5 seconds or if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 5:
        break

# Final capture
ret, final_frame = cap.read()
cap.release()
cv2.destroyAllWindows()

if not ret:
    print("Failed to capture final frame")
    exit()

# Convert captured frame to PIL Image
image = Image.fromarray(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))

# Crop and resize
width, height = image.size
left = 6
top = height / 4
right = 174
bottom = 3 * height / 4

cropped_image = image.crop((left, top, right, bottom))
resized_image = cropped_image.resize((750, 400))

# Convert image to text and speak
img_txt(image)
