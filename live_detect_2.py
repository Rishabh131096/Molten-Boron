import cv2
import os
import numpy as np

subjects = ["Person 1","Sacha Baron Cohen","Person 3"]

def detect_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	if (len(faces) == 0):
		return None, None
 
	(x, y, w, h) = faces[0]

	return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):

	data_folder_path = "training-data"
	dirs = os.listdir(data_folder_path)


	faces = []
	labels = []
 
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue
	
		label = int(dir_name.replace("s", ""))

		subject_dir_path = data_folder_path + "/" + dir_name
 
		subject_images_names = os.listdir(subject_dir_path)

		for image_name in subject_images_names:
			if image_name.startswith("."):
				continue

			image_path = subject_dir_path + "/" + image_name

			image = cv2.imread(image_path)

			face, rect = detect_face(image)

			if face is not None:
				faces.append(face)
				labels.append(label)
 
	return faces, labels
	
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create() 

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
	img = test_img.copy()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	if (len(faces) == 0):
		return img
	
	for f in faces:
		(x, y, w, h) = f
		gray_img = gray[y:y+w, x:x+h]
		rect = (x,y,w,h)
		label,predicted_confidence= face_recognizer.predict(gray_img)
		label_text = subjects[label]
		draw_rectangle(img, rect)
		draw_text(img, label_text, rect[0], rect[1]-5)
	
	return img


cap = cv2.VideoCapture(0)

End_of_Video = False

while(1):
	
	ret, img = cap.read()
	if ret==False:
		End_of_Video = True
		break 
	
	predicted_img = predict(img)
	
	cv2.imshow("VIDEO",predicted_img)
	
	# PRESS ESC TO EXIT
	k = cv2.waitKey(1)
	if k == 27:
		break


cv2.destroyAllWindows()
