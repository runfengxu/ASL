import face_recognition
import os





folders = []

for folder in folders:
	files = os.listdir(folder)
	for file in files:
		img = face_recognition.load_image_file(file)
		locations  = face_recognition.api.face_locations(img,number_of_times_to_unsample=0,model = 'cnn')

		face_number=len(face_location)
		if face_number ==1:
			for face in locations:

				top,right,bottom,left = face_location

		else:
			print('error for folder:',folder,'file: ',file)
