#!/usr/bin/python
# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import re

def get_label(image_path,f):
	content = (f.read())
	# print("Content: ",content)
	# print("Image_path: ",image_path)
	match = re.search((image_path + ".+"),content)

	# print("hmm..: ",match)
	match = match.group() 

	match = match.split(",")[1]
	# match = match.split("\\")
	# print("match: ",match)
	x = match.split("/")[0]
	y = match.split("/")[1]
	# print("X: ",x)
	# print("Y: ",y)
	return x,y

 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# print(f)
for root, dirs, files in os.walk("images"):
	for imagePath in files: 
		print("In image: ",imagePath)
		f = open("LABELS.txt","r")

		expected_x,expected_y = get_label(imagePath,f)
	
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
		image = cv2.imread("images/" + imagePath)
		print("IMG PATH: ",imagePath)
		# image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy()
	 
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
	 
		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 
		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
			# print("x: ",xA)
			# print("y: ",yA)
			# print("expected_x: ",expected_x)
			# print("expected_y: ",expected_y)
			expected_x = int(expected_x)
			expected_y = int(expected_y)
			# print("x+w: ",xA + xB)
			# print("y+h: ",yA + yB)
			x_abs = abs(xA-expected_x)
			y_abs = abs(yA-expected_y)
			if(x_abs < 30 and y_abs < 30):
				print(imagePath + " is a  valid prediction.")
			 
		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))
	 
		# show the output images
		cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)