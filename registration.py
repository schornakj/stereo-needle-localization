'''
Created on Feb 7, 2017

@author: Joe Schornak
'''

import time
import numpy as np
import cv2
import subprocess
import socket
import struct
import argparse
import xml.etree.ElementTree as ET
from collections import deque
import yaml

# Parse commang line arguments. These are primarily flags for things likely to change between runs.
parser = argparse.ArgumentParser(description='Register cameras and phantom to global coordinate frame.')
parser.add_argument('--use_recorded_video', action='store_true',
                    help='Load and process video from file, instead of trying to get live video from webcams.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test',
                    help='Path for video to load if --use_recorded_video is specified.')
args = parser.parse_args()
globals().update(vars(args))

STATE_NO_TARGET_POINTS = 0
STATE_ONE_TARGET_POINT_SET = 1
STATE_SEND_DATA = 2
STATE_NO_DATA = 3

STATE = STATE_NO_TARGET_POINTS


def main():
    global STATE

    # bashCommand = 'mkdir -p ' + output_path
    # process4 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # cv2.waitKey(100)

    if not use_recorded_video:
        # For both cameras, turn off autofocus and set the same absolute focal depth the one used during calibration.
        command = 'v4l2-ctl -d /dev/video1 -c focus_auto=0'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video1 -c focus_absolute=20'
        process1 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video2 -c focus_auto=0'
        process2 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video2 -c focus_absolute=40'
        process3 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)

        # command = 'v4l2-ctl -d /dev/video3 -c focus_auto=0'
        # process5 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)
        # command = 'v4l2-ctl -d /dev/video3 -c focus_absolute=60'
        # process6 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        # cv2.waitKey(100)

        cap_top = cv2.VideoCapture(1)  # Top camera
        cap_side = cv2.VideoCapture(2)  # Side camera
    else:
        # If live video isn't available, use recorded insertion video
        cap_top = cv2.VideoCapture(str(load_video_path + '/output_top.avi'))
        cap_side = cv2.VideoCapture(str(load_video_path + '/output_side.avi'))

    cal_left = Struct(**yaml.load(file('left.yaml','r')))
    cal_right = Struct(**yaml.load(file('right.yaml', 'r')))

    # mat_left_obj = Struct(**cal_left.camera_matrix)
    # mat_left = np.reshape(np.array(mat_left_obj.data),(mat_left_obj.rows,mat_left_obj.cols))



    # mat_right_obj = Struct(**cal_right.camera_matrix)
    # mat_right = np.reshape(np.array(mat_right_obj.data),(mat_right_obj.rows,mat_right_obj.cols))

    mat_left = yaml_to_mat(cal_left.camera_matrix)
    mat_right= yaml_to_mat(cal_right.camera_matrix)
    dist_left = yaml_to_mat(cal_left.distortion_coefficients)
    dist_right = yaml_to_mat(cal_right.distortion_coefficients)

    trans_right = np.array([[-0.0016343138898400025], [-0.13299820438398743], [0.1312384027069722]])
    rot_right = np.array([0.9915492807737206, 0.03743949685116827, -0.12421073976371574, 0.12130773650921836, 0.07179373377171916, 0.9900151982945141, 0.04598322368134065, -0.9967165815148494, 0.06664532446634884]).reshape((3,3))

    p1 = np.concatenate((np.dot(mat_left, np.eye(3)), np.dot(mat_left, np.zeros((3,1)))), axis=1)
    p2 = np.concatenate((np.dot(mat_right, rot_right), np.dot(mat_right, trans_right)), axis=1)

    cv2.namedWindow("Camera Top")
    cv2.namedWindow("Camera Side")

    cv2.setMouseCallback("Camera Top", get_coords_top)
    cv2.setMouseCallback("Camera Side", get_coords_side)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

    # square edge length (m) = 0.0060175

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    print(mat_left)
    print(dist_left)


    while cap_top.isOpened():
        ret, frame_top = cap_top.read()
        ret, frame_side = cap_side.read()
        # ret, aux_frame = cap_aux.read()
        aux_frame = None

        if cv2.waitKey(10) == ord('q') or frame_top is None or frame_side is None:
            break



        frame_top_markers = frame_top
        frame_side_markers = frame_side

        # TODO: Pick three known points in each camera image to define a plane representing the near wall of the phantom
        # TODO: Find the pose of a checkerboard image
        # TODO: solve for the transform between the checkerboard and the origin of the stereo camera pair

        gray = cv2.cvtColor(frame_side, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)


        if ret == True:
            print("Found corners")
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # print("objp", objp)
            # print("corners2", corners2)

            # print(len(objp))
            # print(len(corners2))

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mat_left, dist_left)

            rmat, _ = cv2.Rodrigues(rvecs)
            # print(rmat)
            # print(tvecs*0.0060175)

            transform_homogeneous = np.concatenate((np.concatenate((rmat, tvecs*0.0060175), axis=1), np.array([[0,0,0,1]])), axis=0)
            print(transform_homogeneous)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mat_left, dist_left)

            frame_side_markers = draw(frame_side, corners2, imgpts)
            cv2.imshow('frame_side_markers', frame_side_markers)

        # font = cv2.FONT_HERSHEY_DUPLEX
        # text_color = (0, 255, 0)
        data_frame = np.zeros_like(frame_top)

        # cv2.putText(data_frame, 'Delta: ' + make_data_string(transform_to_robot_coords(delta)),
        #             (10, 50), font, 1, text_color)
        #
        # cv2.putText(data_frame, 'Target: ' + make_data_string(transform_to_robot_coords(position_target)),
        #             (10, 100), font, 1, text_color)
        #
        # cv2.putText(data_frame, 'Tip: ' + make_data_string(transform_to_robot_coords(position_tip)),
        #             (10, 150), font, 1, text_color)
        #
        # cv2.putText(data_frame, 'Top  2D: ' + str(tracker_top.position_tip[0]) + ' ' + str(tracker_top.position_tip[1]),
        #             (10, 200), font, 1, text_color)
        #
        # cv2.putText(data_frame,
        #             'Side 2D: ' + str(tracker_side.position_tip[0]) + ' ' + str(tracker_side.position_tip[1]),
        #             (10, 250), font, 1, text_color)


        combined2 = np.concatenate((data_frame, np.zeros_like(data_frame)), axis=0)

        # if camera_top_with_marker is not None and camera_side_with_marker is not None:
        combined1 = np.concatenate((frame_top_markers, frame_side_markers), axis=0)
        combined = np.array(np.concatenate((combined1, combined2), axis=1), dtype=np.uint8)


        cv2.imshow('Camera Top', frame_top_markers)
        cv2.imshow('Camera Side', frame_side_markers)
        # cv2.imshow("Combined", combined)

    cap_top.release()
    cap_side.release()

    cv2.destroyAllWindows()

class Triangulator:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2

    def _to_float(self, coords):
        return (float(coords[0]), float(coords[1]))

    def get_position_3D(self, coords_top, coords_side):
        pose_3D_homogeneous = cv2.triangulatePoints(self.P1, self.P2,
                                                    np.array(self._to_float(coords_top)).reshape(2, -1),
                                                    np.array(self._to_float(coords_side)).reshape(2, -1))
        return (pose_3D_homogeneous / pose_3D_homogeneous[3])[0:3]

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def yaml_to_mat(input):
    obj = Struct(**input)
    return np.reshape(np.array(obj.data),(obj.rows,obj.cols))

def draw_target_marker(image, target_coords):
    output = image.copy()
    cv2.circle(output, target_coords, 10, (0, 255, 0))
    return output

def get_coords_top(event, x, y, flags, param):
    global STATE
    global TARGET_TOP
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click in top image")
        TARGET_TOP = x, y
        if STATE == STATE_NO_TARGET_POINTS:
            STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
        elif STATE == STATE_ONE_TARGET_POINT_SET:
            STATE == change_state(STATE, STATE_SEND_DATA)
        elif STATE == STATE_SEND_DATA:
            STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)

    elif event == cv2.EVENT_MBUTTONDOWN:
        ESTIMATE_TOP = x, y

def get_coords_side(event, x, y, flags, param):
    global STATE
    global TARGET_SIDE
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click in side image")
        TARGET_SIDE = x, y
        if STATE == STATE_NO_TARGET_POINTS:
            STATE = change_state(STATE, STATE_ONE_TARGET_POINT_SET)
        elif STATE == STATE_ONE_TARGET_POINT_SET:
            STATE == change_state(STATE, STATE_SEND_DATA)
        elif STATE == STATE_SEND_DATA:
            STATE == change_state(STATE, STATE_ONE_TARGET_POINT_SET)

    elif event == cv2.EVENT_MBUTTONDOWN:
        ESTIMATE_SIDE = x, y


def transform_to_robot_coords(input):
    return np.array([-input[2], input[1], -input[0]])

def is_within_bounds(input):
    x_bound = (-60, 80)
    y_bound = (-40, 40)
    z_bound = (70, 210)

    if input[0] >= (x_bound[0] and input[0] <= x_bound[1] and input[1] >= y_bound[0] and input[1] <= y_bound[1]
                    and input[2] >= z_bound[0] and input[2] <= z_bound[1]):
        print('Within bounds!')
        return True
    else:
        return False

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
