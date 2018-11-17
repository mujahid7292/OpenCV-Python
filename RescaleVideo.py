import numpy as np
import cv2

# Lets initialize our webcam
cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)
# If we want to change the resoulation of our webcam
# We will change it before the while loop
#make_480p()
#make_720p()
#make_1080p()
change_res(1000,1900)


# Read every frame from our webcam
while True:
	ret, frame = cap.read()
	# Display those captured frame in our computer.
	cv2.imshow('frame1',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'): 
	# Without this if statement, video will not be shown to the screen
		break

# When everything done release the capture
cap.release()
cv2.destroyAllWindows()