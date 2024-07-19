import gradio as gr
import cv2
import mediapipe as mp



def detect_face(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for processing
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            # Draw connections between landmarks
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]
                pt1 = facial_landmarks.landmark[start_idx]
                pt2 = facial_landmarks.landmark[end_idx]
                x1, y1 = int(pt1.x * width), int(pt1.y * height)
                x2, y2 = int(pt2.x * width), int(pt2.y * height)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

              #Optionally, draw circles on landmarks (can be commented out)
            for i in range(468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 1, (0, 0, 0), -1)

    return image

iface = gr.Interface(fn=detect_face, inputs='image', outputs='image', )
iface.launch(debug=True)
