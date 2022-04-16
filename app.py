import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("My first Streamlit app")
st.write("Hello, world")


def process(img, size, bias):
  # binarize
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, size, bias)
  # fill holes
  contour, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contour:
    cv2.drawContours(bw, [cnt], 0, 255, -1)
  return bw

class VideoProcessor:
    def __init__(self) -> None:
        self.threshold1 = 100
        self.threshold2 = 200

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img, self.threshold1, self.threshold2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
if ctx.video_processor:
    ctx.video_processor.threshold1 = st.slider("Threshold1", min_value=1, max_value=101, step=2, value=51)
    ctx.video_processor.threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=15)
