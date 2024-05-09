import streamlit as st
import requests
import json
import numpy as np
from streamlit_lottie import st_lottie
import streamlit_shadcn_ui as ui
import tempfile
from streamlit_extras.colored_header import colored_header
from streamlit_extras.no_default_selectbox import selectbox
from PIL import Image
from ultralytics import YOLO
import base64,os
import cv2
import plotly.graph_objects as go
import time, math, cvzone
import moviepy.editor as moviepy



st.set_page_config(page_title='Vehicle Logo & Plate Detection',
                   page_icon='🚘',
                   layout='wide',
                   initial_sidebar_state="expanded",)


#Load Model 
logo_model_path = 'models/logo.pt'
plate_model_path = 'models/plate.pt'

car_model = YOLO('yolov8n.pt')
logo_model = YOLO(logo_model_path)
plate_model = YOLO(plate_model_path)

# Functions :

def detect_image(models, image, confidence):
    logo_res = logo_model.predict(image, conf=confidence)
    logo_boxes = logo_res[0].boxes
    logo_names = logo_res[0].names

    plate_res = plate_model.predict(image, conf=confidence)
    plate_boxes = plate_res[0].boxes

    car_res = car_model.predict(image, conf=confidence)
    car_boxes = car_res[0].boxes
    coco_names = car_res[0].names

    # Create an overlay image with bounding boxes and labels
    overlay_image = np.array(image.copy())
    if (len(logo_boxes) > 0) and ("Logo" in models):
        logo_res_plotted = logo_res[0].plot()[:, :, ::-1]
        overlay_image = np.array(logo_res_plotted.copy())
        overlay_image = overlay_image[:, :, ::-1].copy()  # Convert PIL image to NumPy array

    if (len(plate_boxes) > 0) and ("Plate" in models):
        for box in plate_boxes:
            x1, y1, x2, y2 = [box.xyxy[0][i].item() for i in range(4)]
            cv2.rectangle(overlay_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # Draw black background rectangle
            text_size, _ = cv2.getTextSize(f"Plate {box.conf[0]:.2f}", cv2.FONT_HERSHEY_DUPLEX,  0.5, 1)
            cv2.rectangle(overlay_image, (int(x1), int(y2)), (int(x1)+text_size[0], int(y2)+17), (222, 222, 255), -1)
            # Add text
            cv2.putText(overlay_image, f"Plate {box.conf[0]:.2f}", (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    if (len(car_boxes) > 0) and ("Car" in models):
        for box in car_boxes:
            if coco_names[int(box.cls)] == 'car' or coco_names[int(box.cls)] == 'truck' or coco_names[int(box.cls)] == 'bus':
                x1, y1, x2, y2 = [box.xyxy[0][i].item() for i in range(4)]
                cv2.rectangle(overlay_image, (int(x1), int(y1)), (int(x2), int(y2)), (202, 98, 0), 2)
                # Draw black background rectangle
                text_size, _ = cv2.getTextSize(f"{coco_names[int(box.cls)]} {box.conf[0]:.2f}", cv2.FONT_HERSHEY_DUPLEX,  0.5, 1)
                cv2.rectangle(overlay_image, (int(x1), int(y1)), (int(x1)+text_size[0], int(y1)+17), (255, 219, 186), -1)
                # Add text
                cv2.putText(overlay_image, f"{coco_names[int(box.cls)]} {box.conf[0]:.2f}", (int(x1), int(y1) + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (202, 98, 0), 1)

    return overlay_image, logo_boxes, logo_names

def main():
    
    # set title
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json() 

    col1, col2 = st.columns((2,8))
    with col1:      
        url1 = 'https://lottie.host/6d1d2fcc-28ca-4eaa-93ff-5b1e48283852/CoAZZVRFsU.json'
        url2 = 'https://lottie.host/ca8020ae-1881-442a-9aa2-6fba53dcd379/MPb7787v6F.json'
        lottie_animation = load_lottieurl(url2)
        st_lottie(lottie_animation, loop=False, height=200, width=200)
    with col2:
        st.write('##')
        st.write('\n')
        st.markdown("<h1><span style='color:#000000; font-size:45px; font-family:Monaco, Monospace;  font-style: italic;'>Vehicle Logo & Plate Detection</h1>", unsafe_allow_html=True)
    
    # set header
    colored_header(
        label="Welcome to our Object Detection Web App!",
        description=None,
        color_name="blue-green-70",
    )
    
    #set tabs
    tab1, tab2= st.tabs(["App", "About"])

    #Description
    with tab2:
        st.markdown("Our project aims to simplify the task of vehicle logo and Moroccan license plate recognition with cutting-edge AI technology.")
        st.markdown(" Whether you're a developer, a researcher, or simply curious about the capabilities of object detection, our platform offers a user-friendly interface to explore and utilize advanced models, allowing you to detect logos and license plates with precision and efficiency. Plus, you can input your data in various formats (images, videos, or live webcam streams) making the process seamless and adaptable to your workflow.")
        st.markdown("Explore our app to uncover the potential of object detection and discover the extensive list of classes our models can identify below, ranging from popular vehicle logos to Moroccan license plate specific designs. Let's embark on this journey together and unlock the power of AI in recognizing objects in the world around us!")
        st.write('##')
        col1, col2, col3 = st.columns((1,7,1))
        with col2:
            st.image('classes.jpg', caption='24 Classes of Car Brands', width=800)
        st.write('##')
        st.write("© 2024, Developed By TEAMX")

    #App
    with tab1:
        st.write("\n")
        # Model Options
        confidence = float(st.slider("**Select Model Confidence** :", 25, 100, 40)) / 100
        models = st.multiselect("**Select Objects to Detect** :",
                        ["Car", "Logo", "Plate"],
                        ["Car", "Logo", "Plate"])
        choice = selectbox('**Please Select your Input File Type** :',["Upload Image", "Upload Video", "Use Webcam",])


        # Case1: Upload Image
        if choice == 'Upload Image':
            source_img = st.file_uploader(" ", type=['png', 'jpg', 'jpeg', 'bmp', 'webp' ])
            if source_img:
                col1, col2= st.columns(2)
                with col1:
                    uploaded_image = Image.open(source_img)
                    uploaded_image_np = np.asarray(uploaded_image)
                    width, height = uploaded_image.size
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
    
                clicked = ui.button(text="Detect Objects", key="styled_btn_tailwind", className="bg-red-500 text-white font-bold")
                if clicked:
                    with col2:
                        overlay_image, logo_boxes, logo_names = detect_image(models, uploaded_image, confidence)
                        # Display the combined results using st.image
                        st.image(overlay_image,
                            caption="Detected Image",
                            channels='BGR',
                            use_column_width=True)
                        try:
                            with st.expander("Detection Results"):
                                if (len(logo_boxes) > 0) and ("Logo" in models):
                                    for box in logo_boxes:
                                        st.write(f'''-------------------------------------
                                                \nCar : {logo_names[int(box.cls)].capitalize()} 
                                                \nConfidence : {box.conf[0]:.4f}''')
                                        #st.write(box.xywh)
                                        #st.write(box.xyxy)
                                        #st.write(f'''Bounding Box Coordinates : 
                                        #        \nx = {box.xywh[0][0].item()},
                                        #        \ty = {box.xywh[0][1].item()},
                                        #        \nw = {box.xywh[0][2].item()},
                                        #       \th = {box.xywh[0][3].item()}''')
                        except Exception as ex:
                            st.write("No image is uploaded yet!")

        # Case2: Upload Video
        elif choice == 'Upload Video':

            uploaded_video = st.file_uploader("Upload Video", type = ["mp4", "mov",'avi','asf', 'm4v', 'mpeg'])
            if uploaded_video != None:
                col1, col2, col3 = st.columns((1,8,1))
                with col2:
                    vid = uploaded_video.name
                    with open(vid, mode='wb') as f:
                        f.write(uploaded_video.read()) # save video to disk
                    st_video = open(vid,'rb')
                    video_bytes = st_video.read()
                    st.video(video_bytes)
                    #st.write("Uploaded Video")
                
                clicked = ui.button(text="Detect Objects", key="styled_btn_tailwind", className="bg-red-500 text-white font-bold")
                if clicked:
                    col1, col2, col3 = st.columns((1,8,1))
                    with col2:
                        cap = cv2.VideoCapture(vid)

                        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
                        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                        out = cv2.VideoWriter("detected_video.mp4", fourcc, float(frames_per_second), (720, int(720*(9/16))))
                        st_frame = st.empty()
                        
                        while (cap.isOpened()):
                            success, frame = cap.read()
                            if success:
                                frame = cv2.resize(frame, (720, int(720*(9/16))))
                                overlay_image, logo_boxes, logo_names = detect_image(models, frame, confidence)
                                
                                out.write(overlay_image)
                                
                                st_frame.image(overlay_image,
                                    caption='Detected Video',
                                    channels='BGR',
                                    use_column_width=True
                                )
                            else:
                                break
                        cap.release()


        # Case 3 : Use Webcam
        elif choice == 'Use Webcam':
            clicked = ui.button(text="Open Camera", key="styled_btn_tailwind", className="bg-red-500 text-white font-bold")
            if clicked:
                # Loading camera
                cap = cv2.VideoCapture(0)
                cap.set(3, 1280)
                cap.set(4, 720)

                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter("detected_video.mp4", fourcc, 4.0, (int(w), int(h)))
                
                prev_frame_time = 0
                new_frame_time = 0
                st_frame = st.empty()
                while (cap.isOpened()):
                    new_frame_time = time.time()
                    success, img = cap.read()
                    if success:
                        overlay_image, logo_boxes, logo_names = detect_image(models, img, confidence)
                        fps = 1 / (new_frame_time - prev_frame_time)
                        prev_frame_time = new_frame_time
                        cv2.putText(overlay_image, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                    
                        out.write(overlay_image)
                    
                        st_frame.image(overlay_image,
                                    caption='Webcam Detection',
                                    channels='BGR',
                                    use_column_width=True
                                    )
                        key = cv2.waitKey(1)
                    else:
                        break
                cap.release()

if __name__ == "__main__":

    main()


