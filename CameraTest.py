import numpy as np
import cv2

# Lets initialize our webcam
cap = cv2.VideoCapture(0)

# Read every frame from our webcam
while True:
	ret, frame = cap.read()
	# Display those captured frame in our computer.
	cv2.imshow('frame1',frame)
	cv2.imshow('frame2',frame)
	# Create a gray scale display of same video
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray-frame',gray_frame)
	if cv2.waitKey(20) & 0xFF == ord('q'): 
	# Without this if statement, video will not be shown to the screen
		break