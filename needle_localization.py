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

# Parse commang line arguments. These are primarily flags for things likely to change between runs.
parser = argparse.ArgumentParser(description='Do 3D localization of a needle tip using dense optical flow.')
parser.add_argument('--use_connection', action='store_true',
                    help='Attempt to connect to the robot control computer.')
parser.add_argument('--use_recorded_video', action='store_true',
                    help='Load and process video from file, instead of trying to get live video from webcams.')
parser.add_argument('--load_video_path', type=str, nargs=1, default='./data/test',
                    help='Path for video to load if --use_recorded_video is specified.')
parser.add_argument('--save_video', action='store_true',
                    help='Save input and output video streams for diagnostic or archival purposes.')
args = parser.parse_args()
globals().update(vars(args))

# Load xml config file. This is for values that need to be changed but are likely to stay the same for many runs.
tree = ET.parse('config.xml')
root = tree.getroot()
ip_address = str(root.find("ip").text)
port = int(root.find("port").text)
output_dir = str(root.find("output_dir").text)
output_prefix = str(root.find("prefix").text)
hue_target = int(root.find("hue_target").text)
hue_target_range = int(root.find("hue_target_range").text)

TARGET_TOP = (int(258), int(246))
TARGET_SIDE = (int(261), int(230))

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
    global load_video_path

    camera_top_expected_heading = 45
    camera_side_expected_heading = 45

    output_path = output_dir + output_prefix + '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    print(output_path)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if use_connection:
        print('Connecting to ' + ip_address + ' port ' + str(port) + '...')
        s.connect((ip_address, port))

    bashCommand = 'mkdir -p ' + output_path
    process4 = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    cv2.waitKey(100)

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

        command = 'v4l2-ctl -d /dev/video3 -c focus_auto=0'
        process5 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)
        command = 'v4l2-ctl -d /dev/video3 -c focus_absolute=60'
        process6 = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        cv2.waitKey(100)

        cap_top = cv2.VideoCapture(1)  # Top camera
        cap_side = cv2.VideoCapture(2)  # Side camera
    else:
        # If live video isn't available, use recorded insertion video
        cap_top = cv2.VideoCapture(str(load_video_path + '/output_top.avi'))
        cap_side = cv2.VideoCapture(str(load_video_path + '/output_side.avi'))

    cap_aux = cv2.VideoCapture(-1)

    # Load stereo calibration data
    # calibration = np.load('calibration_close.npz')
    calibration = np.load('calibration.npz')

    p1 = calibration['P1']
    p2 = calibration['P2']

    # F = calibration['F']

    # CameraMatrix1 = calibration['CameraMatrix1']
    # DistCoeffs1 = calibration['DistCoeffs1']
    #
    # CameraMatrix2 = calibration['CameraMatrix2']
    # DistCoeffs2 = calibration['DistCoeffs2']

    top_frames = deque(maxlen=3)
    side_frames = deque(maxlen=3)

    ret, camera_top_last_frame = cap_top.read()
    ret, camera_side_last_frame = cap_side.read()

    top_frames.append(camera_top_last_frame)
    side_frames.append(camera_side_last_frame)

    # camera_top_last_frame = cv2.undistort(camera_top_last_frame, CameraMatrix1, DistCoeffs1)
    # camera_side_last_frame = cv2.undistort(camera_side_last_frame, CameraMatrix2, DistCoeffs2)

    # codecArr = 'LAGS'  # Lagarith Lossless Codec
    camera_top_height, camera_top_width, channels = camera_top_last_frame.shape
    camera_side_height, camera_side_width, channels = camera_side_last_frame.shape

    camera_top_roi_size = (200, 350)
    camera_side_roi_size = (200, 350)

    camera_top_roi_center = (int(camera_top_width * 0.8), camera_top_height / 2)

    camera_side_roi_center = (int(camera_side_width * 0.8), camera_side_height / 2)

    delta_last = None
    position_tip_last = None

    trajectory = []

    top_path = []
    side_path = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        filename=output_path + '/output_combined.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(camera_top_width * 2, camera_top_height * 2),
        isColor=True)

    out_top = cv2.VideoWriter(
        filename=output_path + '/output_top.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(camera_top_width, camera_top_height),
        isColor=True)

    out_side = cv2.VideoWriter(
        filename=output_path + '/output_side.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=20.0,
        frameSize=(camera_side_width, camera_side_height),
        isColor=True)

    out_flow = out = cv2.VideoWriter(
        filename=output_path + '/output_flow.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=10.0,
        frameSize=(camera_top_roi_size[0] * 2, camera_top_roi_size[1] * 3),
        isColor=True)

    cv2.namedWindow("Camera Top")
    cv2.namedWindow("Camera Side")

    cv2.setMouseCallback("Camera Top", get_coords_top)
    cv2.setMouseCallback("Camera Side", get_coords_side)

    frames_since_update = 0

    camera_top_farneback_parameters = (0.5, 4, 10, 5, 5, 1.2, 0)
    camera_side_farneback_parameters = (0.5, 4, 10, 5, 5, 1.2, 0)

    tracker_top = TipTracker(camera_top_farneback_parameters, camera_top_width, camera_top_height,
                             camera_top_expected_heading, 40, camera_top_roi_center, camera_top_roi_size)
    tracker_side = TipTracker(camera_side_farneback_parameters, camera_side_width, camera_side_height,
                              camera_side_expected_heading, 40, camera_side_roi_center, camera_side_roi_size)

    target_top = TargetTracker(hue_target, hue_target_range, None)
    target_side = TargetTracker(hue_target, hue_target_range, None)

    triangulator_tip = Triangulator(p1, p2)
    triangulator_target = Triangulator(p1, p2)

    while cap_top.isOpened():
        if cv2.waitKey(10) == ord('q'):
            break

        ret, camera_top_current_frame = cap_top.read()
        ret, camera_side_current_frame = cap_side.read()
        # ret, aux_frame = cap_aux.read()
        aux_frame = None

        top_frames.append(camera_top_current_frame)
        side_frames.append(camera_side_current_frame)

        tracker_top.update(top_frames)
        tracker_side.update(side_frames)

        target_top.update(camera_top_current_frame)
        target_side.update(camera_side_current_frame)

        camera_top_with_marker = draw_tip_marker(camera_top_current_frame, tracker_top.roi_center,
                                                 tracker_top.roi_size, tracker_top.position_tip)
        camera_top_with_marker = draw_target_marker(camera_top_with_marker, target_top.target_coords)

        camera_side_with_marker = draw_tip_marker(camera_side_current_frame, tracker_side.roi_center,
                                                  tracker_side.roi_size, tracker_side.position_tip)
        camera_side_with_marker = draw_target_marker(camera_side_with_marker, target_side.target_coords)

        position_tip = triangulator_tip.get_position_3D(tracker_top.position_tip, tracker_side.position_tip)
        position_target = triangulator_target.get_position_3D(TARGET_TOP, TARGET_SIDE)


        delta = position_target - position_tip
        delta_tform = transform_to_robot_coords(delta)

        if delta_last is not None:
            if not np.array_equal(delta, delta_last):
                frames_since_update = 0
                magnitude = np.linalg.norm(delta - delta_last)
                if magnitude <= MAG_THRESHOLD and frames_since_update <= FRAME_THRESHOLD:
                    SEND_MESSAGES = True
            else:
                SEND_MESSAGES = False
                frames_since_update += 1
            SEND_MESSAGES = True

        if SEND_MESSAGES and not np.array_equal(position_tip, position_tip_last):
            # print('Target: ' + str(target3D))
            # print('Delta: ' + str(delta))
            #
            # print('Target tform: ' + str(transform_to_robot_coords(target3D)))
            # print('Delta tform: ' + str(transform_to_robot_coords(delta)))
            #
            # trajectory.append(transform_to_robot_coords(delta))
            print("Adding point to path")
            top_path.append(tracker_top.position_tip)
            side_path.append(tracker_side.position_tip)

        camera_top_with_marker = draw_tip_path(camera_top_with_marker, top_path)
        camera_side_with_marker = draw_tip_path(camera_side_with_marker, side_path)

        # Send the message to the needle guidance robot controller
        if use_connection and SEND_MESSAGES:
            s.send(compose_OpenIGTLink_message(delta_tform))

        cv2.imshow('Camera Top', camera_top_current_frame)
        cv2.imshow('Camera Side', camera_side_current_frame)

        # cv2.imshow('Camera Top bgr', camera_top_bgr)
        # cv2.imshow('Camera Side bgr', camera_side_bgr)

        font = cv2.FONT_HERSHEY_DUPLEX
        text_color = (0, 255, 0)
        data_frame = np.zeros_like(camera_top_with_marker)

        cv2.putText(data_frame, 'Delta: ' + make_data_string(transform_to_robot_coords(delta)),
                    (10, 50), font, 1, text_color)

        cv2.putText(data_frame, 'Target: ' + make_data_string(transform_to_robot_coords(position_target)),
                    (10, 100), font, 1, text_color)

        cv2.putText(data_frame, 'Tip: ' + make_data_string(transform_to_robot_coords(position_tip)),
                    (10, 150), font, 1, text_color)

        cv2.putText(data_frame, 'Top  2D: ' + str(tracker_top.position_tip[0]) + ' ' + str(tracker_top.position_tip[1]),
                    (10, 200), font, 1, text_color)

        cv2.putText(data_frame,
                    'Side 2D: ' + str(tracker_side.position_tip[0]) + ' ' + str(tracker_side.position_tip[1]),
                    (10, 250), font, 1, text_color)

        if aux_frame is not None:
            combined2 = np.concatenate((data_frame, aux_frame), axis=0)
        else:
            combined2 = np.concatenate((data_frame, np.zeros_like(data_frame)), axis=0)

        out_top.write(camera_top_current_frame)
        out_side.write(camera_side_current_frame)

        # if camera_top_with_marker is not None and camera_side_with_marker is not None:
        combined1 = np.concatenate((camera_top_with_marker, camera_side_with_marker), axis=0)
        combined = np.concatenate((combined1, combined2), axis=1)

        combined_flow = np.concatenate((tracker_top.flow_diagnostic, tracker_side.flow_diagnostic), axis=1)
        cv2.imshow('Combined', combined)
        cv2.imshow('Combined Flow', combined_flow)
        out.write(combined)
        out_flow.write(combined_flow)

        delta_last = delta
        position_tip_last = position_tip

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

    np.savetxt(output_path + "/trajectory.csv", trajectoryArray, delimiter=",")
    np.savez_compressed(output_path + "/trajectory.npz", trajectory=trajectoryArray,
                        top_path=np.array(top_path), side_path=np.array(side_path))


class TipTracker:
    def __init__(self, params, image_width, image_height, heading_expected,
                 heading_range, roi_center_initial, roi_size):
        self.flow_params = params
        self.heading = heading_expected
        self.heading_range = heading_range
        self.roi_center = roi_center_initial
        self.roi_size = roi_size
        self.image_width = image_width
        self.image_height = image_height
        self.position_tip = roi_center_initial

    def _get_section(self, image):
        return image[self.roi_center[1] - self.roi_size[1] / 2:self.roi_center[1] + self.roi_size[1] / 2,
               self.roi_center[0] - self.roi_size[0] / 2:self.roi_center[0] + self.roi_size[0] / 2]

    def _get_dense_flow(self, image_past, image_current):
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image_past, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(image_current, cv2.COLOR_BGR2GRAY),
                                            None,
                                            self.flow_params[0], self.flow_params[1], self.flow_params[2],
                                            self.flow_params[3], self.flow_params[4], self.flow_params[5],
                                            self.flow_params[6])
        flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(image_current)
        hsv[..., 1] = 255

        hsv[..., 0] = (flow_angle * (180 / np.pi) - 90) * 0.5
        hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return hsv, bgr

    def _filter_by_heading(self, flow_hsv):
        min_value = flow_hsv[..., 2].min()
        max_value = flow_hsv[..., 2].max()
        mean_value = flow_hsv[..., 2].mean()

        heading_insert_bound_lower = ((self.heading - self.heading_range / 2) + 180) % 180
        heading_insert_bound_upper = ((self.heading + self.heading_range / 2) + 180) % 180
        flow_hsv_insert_bound_lower = np.array([heading_insert_bound_lower, 50, int(max_value * 0.7)])
        flow_hsv_insert_bound_upper = np.array([heading_insert_bound_upper, 255, max_value])

        mask_insert = cv2.inRange(flow_hsv, flow_hsv_insert_bound_lower, flow_hsv_insert_bound_upper)

        heading_retract_bound_lower = ((self.heading + 90 - self.heading_range / 2) + 180) % 180
        heading_retract_bound_upper = ((self.heading + 90 + self.heading_range / 2) + 180) % 180
        flow_hsv_retract_bound_lower = np.array([heading_retract_bound_lower, 50, int(max_value * 0.7)])
        flow_hsv_retract_bound_upper = np.array([heading_retract_bound_upper, 255, max_value])

        mask_retract = cv2.inRange(flow_hsv, flow_hsv_retract_bound_lower, flow_hsv_retract_bound_upper)

        mask = cv2.bitwise_or(mask_insert, mask_retract)

        kernel = np.ones((7, 7), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilate = cv2.dilate(erosion, kernel, iterations=1)

        ret, thresh = cv2.threshold(dilate, 127, 255, 0)
        return thresh

    def _get_tip_coords(self, image_thresholded):
        position_tip = None

        img, contours, hierarchy = cv2.findContours(image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = []
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areas.append(area)
            contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
            contour_largest = contours_sorted[0][1]

            M = cv2.moments(contour_largest)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            tip_x = self.roi_center[0] - self.roi_size[0] / 2 + cx
            tip_y = self.roi_center[1] - self.roi_size[1] / 2 + cy

            position_tip = (tip_x, tip_y)
        return position_tip

    def _get_new_valid_roi(self, position_tip):
        return (min(max(self.roi_size[0] / 2, position_tip[0]), self.image_width - self.roi_size[0] / 2),
                min(max(self.roi_size[1] / 2, position_tip[1]), self.image_height - self.roi_size[1] / 2))

    def update(self, frames):
        frame_current = frames[-1]
        frame_past = frames[0]
        section_current = self._get_section(frame_current)
        section_past = self._get_section(frame_past)

        self.flow_hsv, self.flow_bgr = self._get_dense_flow(section_past, section_current)

        flow_thresholded = self._filter_by_heading(self.flow_hsv)

        self.flow_diagnostic = np.zeros((2 * self.roi_size[1], self.roi_size[0], 3), np.uint8)
        self.flow_diagnostic[:self.roi_size[1], :, :] = self.flow_bgr
        self.flow_diagnostic[self.roi_size[1]:, :, :] = cv2.cvtColor(flow_thresholded, cv2.COLOR_GRAY2BGR)

        position_tip_new = self._get_tip_coords(flow_thresholded)
        if position_tip_new is not None:
            self.position_tip = position_tip_new
            self.roi_center = self._get_new_valid_roi(self.position_tip)


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

class TargetTracker:
    def __init__(self, target_hsv, target_hsv_range, dims_window):
        self.target_hsv = target_hsv
        self.target_hsv_range = target_hsv_range
        self.dims_window = dims_window

    def update(self, image):
        # TODO: localize target as centroid of cluster near specified HSV values

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        bound_lower = np.array([self.target_hsv/2 - self.target_hsv_range/4, 50, 50])
        bound_upper = np.array([self.target_hsv/2 + self.target_hsv_range/4, 255, 255])

        mask = cv2.inRange(image_hsv, bound_lower, bound_upper)

        kernel = np.ones((7, 7), np.uint8)
        mask_opened = cv2.erode(cv2.dilate(mask, kernel, iterations=1), kernel, iterations=1)

        img, contours, hierarchy = cv2.findContours(mask_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = []
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areas.append(area)
            contours_sorted = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
            contour_largest = contours_sorted[0][1]

            M = cv2.moments(contour_largest)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            self.target_coords = (cx, cy)

def draw_tip_marker(image, roi_center, roi_size, tip_position):
    line_length = 50
    output = image.copy()
    cv2.circle(output, tip_position, 10, (0, 0, 255))
    cv2.rectangle(output, (roi_center[0] - roi_size[0] / 2, roi_center[1] - roi_size[1] / 2),
                  (roi_center[0] + roi_size[0] / 2, roi_center[1] + roi_size[1] / 2), (0, 0, 255), 1)
    # cv2.line(output, tip_position, (int(tip_position[0] - line_length*math.cos(tip_heading)),
    # int(tip_position[1] - line_length*math.sin(tip_heading))), (0,255,0))
    return output


def draw_tip_path(image, path):
    output = image.copy()
    for point in path:
        cv2.circle(output, (int(point[0]), int(point[1])), 7, (80, 127, 255))
    return output


def draw_target_marker(image, target_coords):
    output = image.copy()
    cv2.circle(output, target_coords, 10, (0, 255, 0))
    return output


def get_coords_top(event, x, y, flags, param):
    global STATE
    global TARGET_TOP
    if event == cv2.EVENT_LBUTTONDOWN:
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


def compose_OpenIGTLink_message(input_tform):
    body = struct.pack('!12f', 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                       float(input_tform[0]), float(input_tform[1]), float(input_tform[2]))
    bodysize = 48
    return struct.pack('!H12s20sIIQQ', 1, str('TRANSFORM'), str('SIMULATOR'), int(time.time()), 0, bodysize, 0) + body


def drawlines(img1, line):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, channels = img1.shape
    line = line[0][0]
    # print(line)
    color = (255, 0, 0)
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
    img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
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
