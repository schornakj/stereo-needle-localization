'''
Created on Feb 7, 2017

@author: Joe Schornak
'''

import time
import numpy as np
import cv2
import sys
import getopt
import ffmpy
import math
import subprocess
import socket
import matplotlib.pyplot as plt
import struct
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Do 3D localization of a needle tip using dense optical flow.')
parser.add_argument('--use_connection', action='store_true', help='Attempt to connect to the robot control computer.')
# parser.add_argument('--send_messages', action='store_true', help='Send out OpenIGTLink transforms for the delta between the needle tip and the target.')
parser.add_argument('--use_recorded_video', action='store_true', help='Load and process video from file, instead of trying to get live video from webcams.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test', help='Path for video to load if --use_recorded_video is specified.')
parser.add_argument('--save_video', action='store_true', help='Save input and output video streams for diagnostic or archival purposes.')

args = parser.parse_args()
globals().update(vars(args))

tree = ET.parse('config.xml')
root = tree.getroot()

ip_address = str(root.find("ip").text)
port = int(root.find("port").text)
output_dir = str(root.find("output_dir").text)
output_prefix = str(root.find("prefix").text)

TARGET_TOP = (int(258),int(246))
TARGET_SIDE = (int(261),int(230))

# TARGET_TOP = (int(281),int(228))
# TARGET_SIDE = (int(262),int(272))

ESTIMATE_TOP = (int(200), int(200))
ESTIMATE_SIDE = (int(200), int(200))

SEND_MESSAGES = False

MAG_THRESHOLD = 10
FRAME_THRESHOLD = 5

STATE_NO_TARGET_POINTS = 0
STATE_ONE_TARGET_POINT_SET = 1
STATE_SEND_DATA = 2
STATE_NO_DATA = 3

STATE = STATE_NO_TARGET_POINTS

def main():
	global SEND_MESSAGES
	global STATE

	camera_top_expected_heading = 180
	camera_side_expected_heading = 180

	R_top = 0.16992 # mm/px
	R_side = 0.1864 # mm/px

	offset_px = 36.56

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.axis('equal')

	output_path =  output_dir + output_prefix + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
	print(output_path)

	# ip_address = '192.168.0.103'
	# port = 18944

	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	if use_connection:
		print('Connecting to ' + ip_address + ' port ' + str(port) + '...')
		s.connect((ip_address, port))

	bashCommand = 'mkdir -p ' + output_path
	process4 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	cv2.waitKey(100)


	if not use_recorded_video:
		# For both cameras, turn off autofocus and set the same absolute focal depth the one used during calibration.
		bashCommand = 'v4l2-ctl -d /dev/video1 -c focus_auto=0'
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)
		bashCommand = 'v4l2-ctl -d /dev/video1 -c focus_absolute=20'
		process1 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)
		bashCommand = 'v4l2-ctl -d /dev/video2 -c focus_auto=0'
		process2 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)
		bashCommand = 'v4l2-ctl -d /dev/video2 -c focus_absolute=40'
		process3 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)

		bashCommand = 'v4l2-ctl -d /dev/video3 -c focus_auto=0'
		process5 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)
		bashCommand = 'v4l2-ctl -d /dev/video3 -c focus_absolute=60'
		process6 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		cv2.waitKey(100)
		# bashCommand = 'mkdir /insertion_' + time.strftime("%Y/%m/%d") + '_' + time.strftime("%H:%M:%S")


		cap_top = cv2.VideoCapture(1) # Top camera
		cap_side = cv2.VideoCapture(2) # Side camera
	else:
		# If live video isn't available, use recorded insertion video
		# cap_top = cv2.VideoCapture('./video/close2/video_1.avi')
		# cap_side = cv2.VideoCapture('./video/close2/video_2.avi')
		# cap_top = cv2.VideoCapture('./final insertion 5/output_top.avi')
		# cap_side = cv2.VideoCapture('./final insertion 5/output_side.avi')
		cap_top = cv2.VideoCapture(load_video_path + '/output_top.avi')
		cap_side = cv2.VideoCapture(load_video_path + '/output_side.avi')

	cap_aux = cv2.VideoCapture(-1)

	# Load stereo calibration data
	# calibration = np.load('calibration_close.npz')
	calibration = np.load('calibration.npz')

	P1 = calibration['P1']
	P2 = calibration['P2']

	F = calibration['F']

	CameraMatrix1 = calibration['CameraMatrix1']
	DistCoeffs1 = calibration['DistCoeffs1']

	CameraMatrix2 = calibration['CameraMatrix2']
	DistCoeffs2 = calibration['DistCoeffs2']

	ret, camera_top_last_frame = cap_top.read()
	ret, camera_side_last_frame = cap_side.read()

	# camera_top_last_frame = cv2.undistort(camera_top_last_frame, CameraMatrix1, DistCoeffs1)
	# camera_side_last_frame = cv2.undistort(camera_side_last_frame, CameraMatrix2, DistCoeffs2)

	codecArr = 'LAGS'  # Lagarith Lossless Codec
	camera_top_height, camera_top_width, channels = camera_top_last_frame.shape
	camera_side_height, camera_side_width, channels = camera_side_last_frame.shape

	camera_top_roi_size = (200, 350)
	camera_side_roi_size = (200, 350)
	# camera_top_roi_size = (camera_top_width, camera_top_height)
	# camera_side_roi_size = (camera_side_width, camera_side_height)

	camera_top_roi_center = (int(camera_top_width*0.8),camera_top_height/2)
	camera_top_tip_position = camera_top_roi_center
	# ESTIMATE_TOP = camera_top_roi_center
	camera_top_tip_heading = camera_top_expected_heading

	camera_side_roi_center = (int(camera_side_width*0.8), camera_side_height/2)
	camera_side_tip_position = camera_side_roi_center
	# ESTIMATE_SIDE = camera_side_roi_center
	camera_side_tip_heading = camera_side_expected_heading

	lastDelta = None
	last3DPosition = None

	trajectory = []

	top_path = []
	side_path = []

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	out = cv2.VideoWriter(
		filename=output_path+'/output_combined.avi',
		fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
		fps=20.0,
		frameSize=(camera_top_width*2, camera_top_height*2),
		isColor=True)

	out_top = cv2.VideoWriter(
		filename=output_path+'/output_top.avi',
		fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
		fps=20.0,
		frameSize=(camera_top_width, camera_top_height),
		isColor=True)

	out_side = cv2.VideoWriter(
		filename=output_path+'/output_side.avi',
		fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
		fps=20.0,
		frameSize=(camera_side_width, camera_side_height),
		isColor=True)

	out_flow = 	out = cv2.VideoWriter(
		filename=output_path+'/output_flow.avi',
		fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
		fps=10.0,
		frameSize=(camera_top_roi_size[0]*2, camera_top_roi_size[1]*3),
		isColor=True)

	cv2.namedWindow("Camera Top")
	cv2.namedWindow("Camera Side")

	cv2.setMouseCallback("Camera Top", get_coords_top)
	cv2.setMouseCallback("Camera Side", get_coords_side)

	frames_since_update = 0

	while(cap_top.isOpened()):
		if cv2.waitKey(10) == ord('q'):
			break

		ret, camera_top_current_frame = cap_top.read()
		ret, camera_side_current_frame = cap_side.read()
		ret, aux_frame = cap_aux.read()


		camera_top_height, camera_top_width, channels = camera_top_current_frame.shape

		camera_side_height, camera_side_width, channels = camera_side_current_frame.shape
		camera_top_farneback_parameters = (0.5, 4, 10, 5, 5, 1.2, 0)
		camera_side_farneback_parameters = (0.5, 4, 10, 5, 5, 1.2, 0)
		# scale, levels, window, iterations, poly_n, poly_sigma, flags

		camera_top_new_tip_position, camera_top_new_tip_heading, camera_top_new_roi_center, camera_top_bgr = get_tip_2D_position(camera_top_current_frame, camera_top_last_frame, camera_top_roi_center, camera_top_roi_size, camera_top_expected_heading, camera_top_farneback_parameters)
		camera_top_roi_center = camera_top_new_roi_center

		camera_side_new_tip_position, camera_side_new_tip_heading, camera_side_new_roi_center, camera_side_bgr = get_tip_2D_position(camera_side_current_frame, camera_side_last_frame, camera_side_roi_center, camera_side_roi_size, camera_side_expected_heading, camera_side_farneback_parameters)
		camera_side_roi_center = camera_side_new_roi_center

		# else:
		if camera_top_new_tip_position is not None:
			camera_top_tip_position = camera_top_new_tip_position
		if camera_side_new_tip_position is not None:
			camera_side_tip_position = camera_side_new_tip_position

		if camera_top_new_tip_heading is not None:
				camera_top_tip_heading = camera_top_new_tip_heading
		if camera_side_new_tip_heading is not None:
			camera_side_tip_heading = camera_side_new_tip_heading

		camera_top_with_marker = draw_tip_marker(camera_top_current_frame, camera_top_roi_center, camera_top_roi_size, camera_top_tip_position, camera_top_tip_heading)
		camera_top_with_marker = draw_target_marker(camera_top_with_marker, TARGET_TOP)

		camera_side_with_marker = draw_tip_marker(camera_side_current_frame, camera_side_roi_center, camera_side_roi_size, camera_side_tip_position, camera_side_tip_heading)
		camera_side_with_marker = draw_target_marker(camera_side_with_marker, TARGET_SIDE)

		# triangulatePoints expects matrices of floats, so we need to rebuild the tip coordinates as float tuples instead of int tuples
		camera_top_tip_float = (float(camera_top_tip_position[0]), float(camera_top_tip_position[1]))
		camera_side_tip_float = (float(camera_side_tip_position[0]), float(camera_side_tip_position[1]))

		target_top_float = (float(TARGET_TOP[0]), float(TARGET_TOP[1]))
		target_side_float = (float(TARGET_SIDE[0]), float(TARGET_SIDE[1]))

		# Fancy (but currently troublesome) 3D disparity reconstruction
		tip3D_homogeneous = cv2.triangulatePoints(P1, P2, np.array(camera_top_tip_float).reshape(2,-1), np.array(camera_side_tip_float).reshape(2,-1))
		tip3D = (tip3D_homogeneous/tip3D_homogeneous[3])[0:3]

		target3D_homogeneous = cv2.triangulatePoints(P1, P2, np.array(target_top_float).reshape(2,-1), np.array(target_side_float).reshape(2,-1))
		target3D = (target3D_homogeneous/target3D_homogeneous[3])[0:3]

		delta = target3D - tip3D
		delta_tform = transform_to_robot_coords(delta)

		magnitude = None

		if lastDelta is not None:
			if not np.array_equal(delta, lastDelta):
				frames_since_update = 0
				magnitude = np.linalg.norm(delta - lastDelta)
				if magnitude <= MAG_THRESHOLD and frames_since_update <= FRAME_THRESHOLD: # and is_within_bounds(target3D) and is_within_bounds(tip3D):
					SEND_MESSAGES = True
			else:
				SEND_MESSAGES = False
				frames_since_update+=1
			SEND_MESSAGES = True

		if SEND_MESSAGES and not np.array_equal(tip3D, last3DPosition):
			print('Target: ' + str(target3D))
			print('Delta: ' + str(delta))

			print('Target tform: ' + str(transform_to_robot_coords(target3D)))
			print('Delta tform: ' + str(transform_to_robot_coords(delta)))

			trajectory.append(transform_to_robot_coords(delta))
			top_path.append(camera_top_tip_float)
			side_path.append(camera_side_tip_float)

		camera_top_with_marker = draw_tip_path(camera_top_with_marker, top_path)
		camera_side_with_marker = draw_tip_path(camera_side_with_marker, side_path)

		# Send the message to the needle guidance robot controller
		if use_connection and SEND_MESSAGES:
			s.send(compose_OpenIGTLink_message(delta_tform))

		cv2.imshow('Camera Top',camera_top_current_frame)
		cv2.imshow('Camera Side',camera_side_current_frame)

		cv2.imshow('Camera Top bgr', camera_top_bgr)
		cv2.imshow('Camera Side bgr', camera_side_bgr)

		font = cv2.FONT_HERSHEY_DUPLEX
		text_color = (0,255,0)
		data_frame = np.zeros_like(camera_top_with_marker)
		cv2.putText(data_frame, 'Delta: ' + make_data_string(transform_to_robot_coords(delta)), (10, 50), font, 1, text_color)
		cv2.putText(data_frame, 'Target: ' + make_data_string(transform_to_robot_coords(target3D)), (10, 100), font, 1, text_color)
		cv2.putText(data_frame, 'Tip: ' + make_data_string(transform_to_robot_coords(tip3D)), (10, 150), font, 1, text_color)
		cv2.putText(data_frame, 'Top  2D: ' + str(camera_top_tip_position[0]) + ' ' + str(camera_top_tip_position[1]), (10, 200), font, 1, text_color)
		cv2.putText(data_frame, 'Side 2D: ' + str(camera_side_tip_position[0]) + ' ' + str(camera_side_tip_position[1]), (10, 250), font, 1, text_color)

		if aux_frame is not None:
			combined2 = np.concatenate((data_frame, aux_frame), axis=0)
		else:
			combined2 = np.concatenate((data_frame, np.zeros_like(data_frame)), axis=0)

		out_top.write(camera_top_current_frame)
		out_side.write(camera_side_current_frame)


		if camera_top_with_marker is not None and camera_side_with_marker is not None:
			combined1 = np.concatenate((camera_top_with_marker, camera_side_with_marker), axis=0)
			combined = np.concatenate((combined1, combined2), axis=1)

			combined_flow = np.concatenate((camera_top_bgr, camera_side_bgr), axis=1)
			cv2.imshow('Combined', combined)
			cv2.imshow('Combined Flow', combined_flow)
			out.write(combined)
			out_flow.write(combined_flow)

		camera_top_last_frame = camera_top_current_frame
		camera_side_last_frame = camera_side_current_frame

		lastDelta = delta
		last3DPosition = tip3D

		# cv2.waitKey(0)

	if s is not None:
		s.close()

	cap_top.release()
	cap_side.release()
	out.release()
	out_top.release()
	out_side.release()
	out_flow.release()
	cv2.destroyAllWindows()

	trajectoryArray = np.array(trajectory)
	print(trajectoryArray)

	ax.scatter(trajectoryArray[:,0], trajectoryArray[:,1], trajectoryArray[:,2], c='r', marker='o')

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z (insertion axis)')

	np.savetxt(output_path+"/trajectory.csv", trajectoryArray, delimiter=",")
	np.savez_compressed(output_path+"/trajectory.npz", trajectory=trajectoryArray,top_path=np.array(top_path), side_path=np.array(side_path))

	plt.show()


def get_tip_2D_position(current_frame, last_frame, roi_center, roi_size, expected_heading, p):
	position = None
	heading = None
	new_roi_center = roi_center


	# slice a smaller section of the current frame for optical flow
	current_section = current_frame[roi_center[1]-roi_size[1]/2:roi_center[1]+roi_size[1]/2, roi_center[0]-roi_size[0]/2:roi_center[0]+roi_size[0]/2]

	# slice a section of the last frame that is the same size and in the same position as the new slice
	last_section = last_frame[roi_center[1]-roi_size[1]/2:roi_center[1]+roi_size[1]/2, roi_center[0]-roi_size[0]/2:roi_center[0]+roi_size[0]/2]

	# make a hsv matrix to hold the flow results
	hsv = np.zeros_like(current_section)
	hsv[...,1] = 255

	# convert the sections to grawscale
	prvs = cv2.cvtColor(last_section,cv2.COLOR_BGR2GRAY)
	next = cv2.cvtColor(current_section,cv2.COLOR_BGR2GRAY)

	# conduct optical flow with some generic parameters and write the results to hsv
	# hue is direction, value is magnitude
	flow = cv2.calcOpticalFlowFarneback(prvs,next, None, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
		# scale, levels, window, iterations, poly_n, poly_sigma, flags
#		 (0.5, 3, 6, 5, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	min_value = hsv[..., 2].min()
	max_value = hsv[..., 2].max()
	mean_value = hsv[..., 2].mean()

	heading_insert_lower_bound = ((expected_heading - 20)/2 + 180)%180
	heading_insert_upper_bound = ((expected_heading + 20)/2 + 180)%180
	insert_lower_bound= np.array([heading_insert_lower_bound, 50, int(max_value*0.7)])
	insert_upper_bound = np.array([heading_insert_upper_bound, 255, max_value])
	mask_insert = cv2.inRange(hsv, insert_lower_bound, insert_upper_bound)

	heading_retract_lower_bound = ((expected_heading + 90 - 10)/2 + 180)%180
	heading_retract_upper_bound = ((expected_heading + 90 + 10)/2 + 180)%180
	retract_lower_bound= np.array([heading_retract_lower_bound, 50, int(max_value*0.7)])
	retract_upper_bound = np.array([heading_retract_upper_bound, 255, max_value])
	mask_retract = cv2.inRange(hsv, retract_lower_bound, retract_upper_bound)

	# mask = cv2.bitwise_or(mask_insert, mask_retract)
	mask = mask_insert

	kernel = np.ones((7,7),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	dilate = cv2.dilate(erosion,kernel,iterations = 1)

	ret,thresh = cv2.threshold(dilate,127,255,0)

	img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	diagnostic = np.zeros((3*roi_size[1],roi_size[0],3), np.uint8)
	diagnostic[:roi_size[1], :, :] = bgr
	diagnostic[roi_size[1]:2*roi_size[1], :, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	diagnostic[2*roi_size[1]:, :, :] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)



	if len(contours)>0:
			areaArray = []
			for i, c in enumerate(contours):
				area = cv2.contourArea(c)
				areaArray.append(area)
			sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
			largestcontour = sorteddata[0][1]

			M = cv2.moments(largestcontour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])

			# print('X: ' + str(cx) + " Y: " + str(cy))

			newX = roi_center[0] - roi_size[0]/2 + cx
			newY = roi_center[1] - roi_size[1]/2 + cy

			height, width, channels = current_frame.shape
			position = (newX, newY)

			edges = cv2.Canny(current_section,100,200)
			minLineLength = current_section.shape[1]-1
			maxLineGap = 10
			lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)

			if lines is not None:
				lengthArray=[]
				for i, line in enumerate(lines):
					length = math.sqrt(pow(line[0][2]-line[0][0],2)+pow(line[0][3]-line[0][1],2))
					lengthArray.append(length)
				sortedlines = sorted(zip(lengthArray,lines), key = lambda x: x[0], reverse=True)
				longestline = sortedlines[0][1]

				heading = math.atan2(longestline[0][3]-longestline[0][1], longestline[0][2]-longestline[0][0])

			# if newX < width - roi_size[0]/2 and newX > roi_size[0]/2 and newY < height - roi_size[1]/2 and newY > roi_size[1]/2:
			# 	new_roi_center = position

			# if position is not None:
			# Constrain the new ROI center so the bounding box does not go beyond the image borders
			new_roi_center = (min(max(roi_size[0]/2, newX), width - roi_size[0]/2), min(max(roi_size[1]/2, newY), height - roi_size[1]/2))


	return position, heading, new_roi_center, diagnostic

def draw_tip_marker(image, roi_center, roi_size, tip_position, tip_heading):
	line_length = 50
	output = image.copy()
	cv2.circle(output, tip_position, 10, (0,0,255))
	cv2.rectangle(output, (roi_center[0]-roi_size[0]/2,roi_center[1]-roi_size[1]/2), (roi_center[0]+roi_size[0]/2,roi_center[1]+roi_size[1]/2), (0,0,255), 1)
	# cv2.line(output, tip_position, (int(tip_position[0] - line_length*math.cos(tip_heading)), int(tip_position[1] - line_length*math.sin(tip_heading))), (0,255,0))
	return output

def draw_tip_path(image, path):
	output = image.copy()
	for point in path:
		cv2.circle(output, (int(point[0]), int(point[1])), 7, (80,127,255))
	return output

def draw_target_marker(image, target_coords):
	output = image.copy()
	cv2.circle(output, target_coords, 10, (0,255,0))
	return output

def get_coords_top(event, x, y, flags, param):
	global STATE
	global TARGET_TOP
	if event == cv2.EVENT_LBUTTONDOWN:
		TARGET_TOP = x,y
		if STATE == STATE_NO_TARGET_POINTS:
			STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
		elif STATE == STATE_ONE_TARGET_POINT_SET:
			STATE == change_state(STATE, STATE_SEND_DATA)
		elif STATE == STATE_SEND_DATA:
			STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)

	elif event == cv2.EVENT_MBUTTONDOWN:
		ESTIMATE_TOP = x,y

def get_coords_side(event, x, y, flags, param):
	global STATE
	global TARGET_SIDE
	if event == cv2.EVENT_LBUTTONDOWN:
		TARGET_SIDE = x,y
		if STATE == STATE_NO_TARGET_POINTS:
			STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
		elif STATE == STATE_ONE_TARGET_POINT_SET:
			STATE == change_state(STATE, STATE_SEND_DATA)
		elif STATE == STATE_SEND_DATA:
			STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)

	elif event == cv2.EVENT_MBUTTONDOWN:
		ESTIMATE_SIDE = x,y

def linear_to_3D(coords_top, coords_side, R_top, R_side, offset_px):
		x_mm = -coords_top[1]*R_top
		y_mm = coords_side[1]*R_side
		z_mm = (coords_top[0]*R_top + (coords_side[0]+offset_px)*R_side)/2

		return np.array([x_mm, y_mm, z_mm])

def transform_to_robot_coords(input):
	return np.array([-input[2], input[1], -input[0]])

def is_within_bounds(input):
	x_bound = (-60, 80)
	y_bound = (-40, 40)
	z_bound = (70, 210)

	if input[0] >= x_bound[0] and input[0] <= x_bound[1] and input[1] >= y_bound[0] and input[1] <= y_bound[1] and input[2] >= z_bound[0] and input[2] <= z_bound[1]:
		print('Within bounds!')
		return True
	else:
		return False

def compose_OpenIGTLink_message(input_tform):
	body = struct.pack('!12f', 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, float(input_tform[0]), float(input_tform[1]), float(input_tform[2]))
	bodysize = 48
	return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body

def drawlines(img1,line):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, channels = img1.shape
    line = line[0][0]
    # print(line)
    color = (255,0,0)
    x0,y0 = map(int, [0, -line[2]/line[1] ])
    x1,y1 = map(int, [c, -(line[2]+line[0]*c)/line[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    return img1

def change_state(current_state, new_state):
	if current_state == STATE_NO_TARGET_POINTS:
		if new_state == STATE_ONE_TARGET_POINT_SET:
			return new_state

	elif current_state == STATE_ONE_TARGET_POINT_SET:
		if new_state == STATE_SEND_DATA or new_state == STATE_NO_TARGET_POINTS:
			return new_state

	elif current_state == STATE_SEND_DATA:
		if new_state == STATE_NO_DATA or new_state == STATE_ONE_TARGET_POINT_SET:
			return new_state

	elif current_state == STATE_NO_DATA:
		if new_state == STATE_SEND_DATA or new_state == STATE_ONE_TARGET_POINT_SET:
			return new_state

	else:
		return current_state

def print_state(current_state):
	if current_state == STATE_NO_TARGET_POINTS:
		print('STATE_NO_TARGET_POINTS')
	elif current_state == STATE_ONE_TARGET_POINT_SET:
		print('STATE_ONE_TARGET_POINT_SET')
	elif current_state == STATE_SEND_DATA:
		print('STATE_SEND_DATA')
	elif current_state == STATE_NO_DATA:
		print('STATE_NO_DATA')

def make_data_string(data):
	return '%0.3g, %0.3g, %0.3g' % (data[0], data[1], data[2])



if __name__ == '__main__':
	main()
