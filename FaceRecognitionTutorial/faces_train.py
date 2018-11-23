import os
from PIL import Image # Python image processing library
import numpy as np
import cv2
import pickle

# Now we will get 'face_train.py' file's base dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
# This above command will print following directory in our terminal
# D:\GoogleDrive\TUTORIALS\AI Tutorial\OpenCVAndPython\FaceRecognitionTutorial
image_dir = os.path.join(BASE_DIR,'images')
# print(image_dir)
# This above command will print following directory in our terminal
# D:\GoogleDrive\TUTORIALS\AI Tutorial\OpenCVAndPython\FaceRecognitionTutorial\images

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')


# Creating a label with number value
current_id = 0 # This is the first training person's id
# Create an empty dictionary for our label's id
label_ids = {} # {'mujahid':0,'emilla_clarke':1}


y_labels = []
x_train = []

# Now we want to loop through those images
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			# Code execution enter this line of code, that means
			# our file is an image. So now we will print the files dir
			img_file_path = os.path.join(root,file)
			# We will grab the name of the folder and mark it as 'img_label'
			# img_label = os.path.basename(os.path.dirname(img_file_path)).replace(" ", "-").lower()
			# This below line and above lines are the same
			img_label = os.path.basename(root).replace(" ", "-").lower()
			# print(img_label, img_file_path)
			# print(img_label) # This will print 'mujahid'....
			# Now we will create a dictionary of those label
			if not img_label in label_ids:
				# We will insert those label in the 'label_ids' dictionary
				label_ids[img_label] = current_id
				#         'mujahid'  :  0
				# Now we will increase the current id by one number
				current_id += 1
			# Now we will save current person's numerical id in 'id_' variable
			id_ = label_ids[img_label]
			print('{} : {}'.format(img_label, id_))
			# Now we will open that image with python image processing library.
			# Then convert this image into grayscale by convert('L') method
			pil_image = Image.open(img_file_path).convert("L")
			# Now we will resize all of those image to same size
			# image_size = (550,550)
			# final_image = pil_image.resize(image_size,Image.ANTIALIAS)
			# Now we will convert this gray image into numpy array
			# image_array = np.array(final_image, 'uint8') # Does Not Work
			image_array = np.array(pil_image, 'uint8')
			# Now we will print this numpy image array
			# print(image_array)
			# We will not train our model on the whole image. We will train our model
			# only on people's face's
			# Now we will find all the face's in this numpy 'image_array' & save those faces in
			# faces variable
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors=5)
			# scaleFactor=1.5 -> Will scale our face by 1.5 times
			# Now we will iterate through those face's
			for (x, y, w, h) in faces:
				# print(x,y,w,h)        #[Y_Start:Y_End, X_Start:X_End]
				region_of_interest = image_array[y:y+h, x:x+w] # This will create square image
				# Now we will save all the face / roi in the training array
				x_train.append(region_of_interest)
				# So in above line, we have preprared our training data
				# But we don't know whose face we are training our model on?
				# So we have to provide label with our training data
				y_labels.append(id_)
				# Now we will save those 'y_labels'. So that we can use this labeled face's in
				# faces.py file 


# Now lets save those 'y_label'
with open('labels.pickle', 'wb') as file:
	pickle.dump(label_ids, file)
	# So we are going to dump those 'label_ids' in this 'file' 


# wb = As we are writing bytes

# Now we will train our model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(x_train, np.array(y_labels))
face_recognizer.save('trainner.yml')
