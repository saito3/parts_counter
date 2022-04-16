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

  D = ndimage.distance_transform_edt(bw)
  localMax = peak_local_max(D, indices=False, min_distance=20,
	  labels=bw)
  markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
  labels = watershed(-D, markers, mask=bw)

  count = 0
  for label in np.unique(labels):
    if label == 0:
      continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    if r < 10:
      continue
    count += 1
    cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(img, "#{}".format(count), (int(x) - 10, int(y)),
	  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
  return img

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

cv2_imshow(-D/100)

class VideoProcessor:
  def __init__(self) -> None:
    self.size = 100
    self.bias = 200

  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    img = process(img, self.size, self.bias)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
if ctx.video_processor:
  ctx.video_processor.size = st.slider(
    "Size", min_value=81, max_value=21, step=2, value=51)
  ctx.video_processor.bias = st.slider(
    "Bias", min_value=0, max_value=30, step=1, value=15)
