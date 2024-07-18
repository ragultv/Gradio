import gradio as gr
from ultralytics import YOLO

#load the model
model=YOLO('yolov8n.pt')
#define the detection 
def detect(img):
    results=model(img)
    return results[0].plot()

iface=gr.Interface(fn=detect,inputs='image',outputs='image',)
#launch the interface
iface.launch()
    
