import gradio as gr
import cv2
from retinaface import RetinaFace
#define the detect
def detect(img):
    results = RetinaFace.detect_faces(img)
    # Check if any faces were detected
    if results:
        image = img.copy()
        for key in results:
            face = results[key]
            facial_area = face['facial_area']
            # Draw a rectangle around the detected face
            x1, y1, x2, y2 = facial_area
            img_with_face = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return image


iface = gr.Interface(fn=detect, inputs='image', outputs='image', )
#launch the interface
iface.launch(debug=True)
