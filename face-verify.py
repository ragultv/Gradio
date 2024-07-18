import gradio as gr
from ultralytics import YOLO
from deepface import DeepFace
#load the model
model=YOLO('yolov8n.pt')
#define the function
def detect(img1,img2):
    results=DeepFace.verify(img1,img2)
    if results['verified']:
        return 'Verified'
    else:
        return 'Not Verified'

iface=gr.Interface(fn=detect,inputs=['image','image'],outputs='text',)
#launch the interface
iface.launch()
    
