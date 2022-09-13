import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
from pytube import YouTube
import os
import pyglet
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
#singer = st.text_input("singer")

if lang  and st.session_state["run"] !="true":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		#webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song")
                op = webdriver.ChromeOptions()
                op.add_argument('headless')
                driver = webdriver.Chrome(options=op)
                driver.get(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song")
                select = driver.find_element(By.CSS_SELECTOR, 'div#contents ytd-item-section-renderer>div#contents a#thumbnail')
                link = [select.get_attribute('href')]
                yt = YouTube(link[0])
                video = yt.streams.filter(only_audio=True).first()
                out_file = video.download(output_path="C:/Users/Admin/Desktop/music")
                base, ext = os.path.splitext(out_file)
                new_file = base + '.mp3'
                os.rename(out_file, new_file)
                width = 500
                height = 500
                title = "Music Player"
                window = pyglet.window.Window(width, height, title)

                vidPath =new_file.split("\\")[1]

                # creating a media player object
                player = pyglet.media.Player()

                # creating a source object
                source = pyglet.media.StreamingSource()

                # load the media from the source
                MediaLoad = pyglet.media.load(vidPath)

                # add this media in the queue
                player.queue(MediaLoad)

                # play the video
                player.play()

                # on draw event
                @window.event
                def on_draw():
                        
                        # clear the window
                        window.clear()
                        
                        # if player source exist
                        # and video format exist
                        if player.source and player.source.video_format:
                                
                                # get the texture of video and
                                # make surface to display on the screen
                                player.get_texture().blit(0, 0)
                                
                                
                # key press event	
                @window.event
                def on_key_press(symbol, modifier):
                        if symbol == pyglet.window.key.P:

                            # pause the video
                            player.pause()

                            # printing message
                            print("Video is paused")


                        # key "r" get press
                        if symbol == pyglet.window.key.R:

                            # resume the video
                            player.play()

                            # printing message
                        
                pyglet.app.run()

                np.save("emotion.npy", np.array([""]))
                st.session_state["run"] != "false"
