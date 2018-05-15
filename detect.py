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
import operator

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def get_label(image_path,f):
	content = (f.read())
	match = re.search((image_path + ".+"),content)

	match = match.group() 

	label = match.split(",")

	x1 = label[1].split("/")[0]
	y1 = label[1].split("/")[1]
	x2 = label[2].split("/")[0]
	y2 = label[2].split("/")[1]

	return x1,y1,x2,y2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# print(f)
im_accuracy = {}
for root, dirs, files in os.walk("images"):
	for imagePath in files: 
		print("In image: ",imagePath)
		f = open("LABELS.txt","r")

		expected_x1,expected_y1,expected_x2,expected_y2 = get_label(imagePath,f)
		expected_x1 = int(expected_x1)
		expected_y1 = int(expected_y1)
		expected_x2 = int(expected_x2)
		expected_y2 = int(expected_y2)
		#load pre-processed images
		image = cv2.imread("images/" + imagePath)
		print("IMG PATH: ",imagePath)
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 
		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
			boxA = [xA, yA, xB, yB]
			boxB = [expected_x1, expected_y1, expected_x2, expected_y2]
			accuracy = bb_intersection_over_union(boxA, boxB)
			print("accuracy:", accuracy)
			im_accuracy[imagePath] = accuracy
			 
		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))
		# show the output images
		cv2.rectangle(image, (expected_x1, expected_y1), (expected_x2, expected_y2), (255, 0, 0), 2)
		cv2.imshow("Predicted (Green) with Expected (Blue)", image)
		cv2.waitKey(0)

avg_accuracy = 0
for key in im_accuracy.keys():
	avg_accuracy += im_accuracy[key]
avg_accuracy = avg_accuracy/len(im_accuracy)
print("Average Accuracy: ", avg_accuracy)

sorted_im = sorted(im_accuracy.iteritems(), key=operator.itemgetter(1))
top_3 = sorted_im[len(sorted_im)-3:len(sorted_im)].reverse()
print(top_3)