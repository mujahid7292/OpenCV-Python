import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def reScaleFrame(frame, percent=75):
	# Default percent is 75%
	width = int(frame.shape[1] * percent / 100)
	height = int(frame.shape[0] * percent / 100)
	dimension = (width, height)
	return cv2.resize(frame, dimension, interpolation =cv2.INTER_AREA)


# Read every frame from our webcam
while True:
	ret, frame = cap.read()
	# Now create 2 different frame with 2 different size
	frame1 = reScaleFrame(frame,percent=50)
	frame2 = reScaleFrame(frame,percent=100)
	# Display those 2 frame in our computer.
	cv2.imshow('frame1',frame1)
	cv2.imshow('frame2',frame2)
	if cv2.waitKey(20) & 0xFF == ord('q'): 
	# Without this if statement, video will not be shown to the screen
		break

# When everything done release the capture
cap.release()
cv2.destroyAllWindows()