import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# Grab our trained data "trainner.yml"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainner.yml')
Faces_Labels = {}
# Now grab those saved 'y_label'
with open('labels.pickle', 'rb') as file:
	Original_Faces_Labels = pickle.load(file)
	# But problem here is that we have saved our faces label like {'mujahid':3}
	# But we don't want the id value, we want corrosponding names value. So we 
	# have to reverse this dictionary {3:'mujahid'}
	Faces_Labels = {v:k for k,v in Original_Faces_Labels.items()}

cap = cv2.VideoCapture(0)

while True:
	# Read video from cap
	ret, frame = cap.read()
	# Create a gray frame. Haarcascade can not detect face's without gray frame
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# Now we will find all the face's in this gray_frame & save those faces in faces variable
	faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5,minNeighbors=3)
	# scaleFactor=1.5 -> Will scale our face by 1.5 times
	#Now we will find all the eye in our frame
	eyes = eye_cascade.detectMultiScale(gray_frame)
	# Now we will iterate through those face's
	for (x, y, w, h) in faces:
		# print(x,y,w,h)                     #[Y_Start:Y_End, X_Start:X_End]
		region_of_interest_in_gray_frame = gray_frame[y:y+h, x:x+w] # This will create square image
		region_of_interest_in_color_frame =frame[y:y+h, x:x+w]
		# Below we will predict the faces based on "trainner.yml"
		id_, confidence = face_recognizer.predict(region_of_interest_in_gray_frame)
		# This confidence interval create problem it goes above 100 & below 0
		# So we will predict our face if confidence interval is in certain threshhold
		if confidence >= 45 and confidence <= 85:
			print("Name: {} | ID: {} | Confidence: {}".format(Faces_Labels[id_], id_, confidence))
			# Now we will show the predicted person's name on the screen
			font = cv2.FONT_HERSHEY_SIMPLEX
			font_size = 1
			name = Faces_Labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, font_size, color, stroke, cv2.LINE_AA)
		# Now we will take a snapshot of our roi
		snapshot_on_gray = 'snapshot_gray.png'
		snapshot_on_color = 'snapshot_color.png'
		# Take Snapshot
		cv2.imwrite(snapshot_on_gray,region_of_interest_in_gray_frame)
		cv2.imwrite(snapshot_on_color,region_of_interest_in_color_frame)
		# Now we will create a blue color rectangle around our faces
		face_box_color = (255,0,0) # BGR = Blue, Green, Red
		# Stroke of the rectangle box
		stroke = 2
		# End Coordinate of X of the face rectangle
		end_cord_x = x + w
		# End Coordinate of Y of the face rectangle
		end_cord_y = y + h
		# Now we will write this rectangle on our color frame
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), face_box_color, stroke)
		# Now we iterate through those eye in the frame
		for (ex, ey, ew, eh) in eyes:
			# Now we will put a rectangle around our eye's
			cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0,255,0), 1)

	# Show video on the screen
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord("q"):
		# Without this if statement, video will not be shown to the screen
		break

cap.release()
cv2.destroyAllWindows()
