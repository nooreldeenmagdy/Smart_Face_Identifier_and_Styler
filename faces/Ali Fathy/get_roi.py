import cv2 as cv
import os
import numpy as np

def is_image_file(file_name):
    image_file_extensions = ('.rgb', '.gif', '.pbm', '.pgm', '.ppm', '.tiff', '.rast' '.xbm',
    	'.jpeg', '.jpg', '.JPG', '.bmp', '.png', '.PNG', '.webp', '.exr')
    return file_name.endswith((image_file_extensions))

face_cascade = cv.CascadeClassifier('../../cascades/haarcascade_frontalface_alt.xml')
with open('roi-names', 'rt') as f:
	roi_files = f.readlines()
	roi_files = [i.split('\n')[0] for i in roi_files]

print(roi_files)

for root, dirs, files in os.walk('.'):
	for file in files:
			if is_image_file(file) and file not in roi_files:
				path = os.path.join(root, file)
				print(file)

				img = cv.imread(file)
				img = cv.resize(img, (540, 540), interpolation = cv.INTER_AREA)
				gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

				faces = face_cascade.detectMultiScale(gray,
													scaleFactor=1.1,
													minNeighbors=5,
													minSize=(50, 50),
													flags=cv.CASCADE_SCALE_IMAGE)
				for (x,y,w,h) in faces:
					roi = img[y: y+h, x: x+w]
					roi_file = 'roi-{}'.format(file)

					cv.imwrite(roi_file, roi)
					with open('roi-names', 'at') as f:
						f.write(file+'\n')
						f.write(roi_file+'\n')
