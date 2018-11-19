import os
import numpy as np 
import cv2

# Let's first set a name for our video file
video_file_name = 'video.avi'
frames_per_second = 24.0
my_resolution = '480p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Create a python dictionary consist of Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_video_dim(cap,resolution='1080p'):
	# First set default height & width of 480p
	width, height = STD_DIMENSIONS['480p']
	# Now we will check whether our passed resoulation is
	# in the dictionary or not
	if resolution in STD_DIMENSIONS:
		width, height = STD_DIMENSIONS[resolution]
	change_res(cap,width,height)
	return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Create a capture variable to store videoCapture 
capture = cv2.VideoCapture(0)
# Now get our videos dimension
dimension = get_video_dim(capture,resolution = my_resolution)
# Now we will get video type mp4, avi
video_type_cv2 = get_video_type(video_file_name)

# We will save our video in 'output' variable
output = cv2.VideoWriter(video_file_name, video_type_cv2, frames_per_second, dimension)

while True:
	# Capture video frame by frame
	ret, frame = capture.read()
	# Now we will write video on 'output' variable
	output.write(frame)
	# Display the resulting frame
	cv2.imshow('Frame',frame)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		# Without this if statement, video will not be shown to the screen
		break

# When everything done release the capture
capture.release()
output.release()
cv2.destroyAllWindows()

