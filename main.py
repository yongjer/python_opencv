import cv2
from cv2 import dnn_superres
# Create an SR object - only function that differs from c++ code
sr = dnn_superres.DnnSuperResImpl_create()
# Read image
video = cv2.VideoCapture("/Users/yongjer/Downloads/P1 Squences of real numbers & the axiom of completeness.mp4")
# Read the desired model
path = "./EDSR_x2.pb"
sr.readModel(path)
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter('/Users/yongjer/Downloads/output.mp4', fourcc, 30, (1440, 960))
while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    else:
        result = sr.upsample(frame)
        out.write(result)


