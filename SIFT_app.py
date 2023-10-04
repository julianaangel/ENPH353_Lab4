#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi #allows us to convert a QtDesigner .ui file into Python Object

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	#gets live stream to display in the live_image_label when the "Enable camera" button is pressed	
	def __init__(self):

		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self) #contructor loads the .ui file

		self._cam_id = 1
		self._cam_fps = 24
		self._is_cam_enabled = False
		self._is_template_loaded = False

		#connect button "clicked" signal to brouse_button slot function
		self.browse_button.clicked.connect(self.SLOT_browse_button)
		#register click event from toggle camera button
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		#create cv2 camera object with 320x240 resolution
		self._video_path = "./IMG_4412.mp4"
		self._camera_device = cv2.VideoCapture(self._video_path)
		# self._camera_device.set(3, 320)
		# self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		#emits a signal every time the set interval elapses
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)#connected timer signnal to slot_query functio
		self._timer.setInterval(1000 / self._cam_fps)

	#add the slot function to class
	def SLOT_browse_button(self):
		#dlg instantiates a File Dialog object
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		#run the file dialog object, blocks the app event loop
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	#captures a frame from camera every time timer interval elapses
	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		#TODO run SIFT on the captured frame
		target_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)#image we're looking for

		# Features detecting, load algorthm
		sift = cv2.xfeatures2d.SIFT_create()
		kp_image, descriptor_image= sift.detectAndCompute(target_img,None)#key points on image

		#need to detect featutres
		#algorithm detects and matches features
		index_params = dict(algorithm = 0, trees=5)#dictionary
		search_params = dict()
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		kp_grayframe, descriptor_grayframe = sift.detectAndCompute(grayframe,None)

		#k nearest neighbours - for every point in the image we find 
		matches = flann.knnMatch(descriptor_image,descriptor_grayframe,k=2)#finds matches: compares descriptors
		#have a point in template, and you have 2 other points that look very similar, this is hard to work with
		#only want distinct points
		
		unique_points = []
		for m, n in matches:#m is query image, n is object from train image (grayframe)
			if m.distance < 0.6*n.distance:
				#want to take m when it is much shorter than n because good points are the ones that are most unique
				#distance is distance between descriptor of match and descriptor of keypoint
				#distance between vectors that represent colour variants
				#when you decrease the coefficient we get less false matches
				unique_points.append(m)

		

		#homography
		if len(unique_points) > 7:#if we find at least 7 matches, draw homogtaphy
			#extracting position of points of the query image
			query_pts = np.float32([kp_image[m.queryIdx].pt for m in unique_points]).reshape(-1,1,2)#format that we want to extract in
			#query_idx gives us position of points in query image
			#reshape is the way we change the form of the array
			train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in unique_points]).reshape(-1,1,2)

			matrix, mask = cv2.findHomography(query_pts,train_pts,cv2.RANSAC, 5.0)#matrix can let us show the object in its perspective
			# RANSAC fits models in the presence of many data outliers
			#agrees on the linear transformation
			matches_mask = mask.ravel().tolist()

			#perspective transform
			height, width = target_img.shape
			pts = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,matrix)#passing the points with the height and width of original image
			
			#now we can draw the lines on object detected
			display = cv2.polylines(frame,[np.int32(dst)], True, (255,0,0),3)#pixel must be integer number

			pixmap = self.convert_cv_to_pixmap(display)
			#close lines is true, blue lines, thickness 3
		#end of code

		else:
			#create a frame large enough to have the target image and the live frame side by side with good matches drawn

			bigframe = np.zeros((max(target_img.shape[0],grayframe.shape[0]), target_img.shape[1]+grayframe.shape[1], 3), np.uint8)
			bigframe = cv2.drawMatches(target_img,kp_image,frame, kp_grayframe,unique_points,frame)
			pixmap = self.convert_cv_to_pixmap(bigframe)

		self.live_image_label.setPixmap(pixmap)
	#turns timer on or off(and changes lable on button)
	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

#instantiate myApp class
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_()) #starts Qt event loop, which runs until we close the progran
