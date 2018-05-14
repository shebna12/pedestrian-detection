# import the necessary packages
from __future__ import print_function
import os
import imutils
import cv2
 
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

filenames = []
for root, dirs, files in os.walk("images"):
	for image_path in files: 
		filenames.append(image_path)

print(filenames)
# loop over the image paths
for imagename in filenames:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy

	image = cv2.imread("images"+ "/" + imagename)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	# orig = image.copy()
	cv2.imwrite("images/" + imagename,image)
	cv2.waitKey(0)