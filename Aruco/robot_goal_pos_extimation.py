#!/usr/bin/env python3

from __future__ import print_function
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import sys

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.13

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def calculate_robot_position(marker_positions):
    """
    Calculate the robot's position based on detected markers 1 and 2.
    """
    if 1 in marker_positions and 2 in marker_positions:
        x1, y1, z1 = marker_positions[1]
        x2, y2, z2 = marker_positions[2]
        robot_x = (x1 + x2) / 2
        robot_y = (y1 + y2) / 2
        robot_z = (z1 + z2) / 2
    elif 1 in marker_positions:
        robot_x, robot_y, robot_z = marker_positions[1]
    elif 2 in marker_positions:
        robot_x, robot_y, robot_z = marker_positions[2]
    else:
        robot_x, robot_y, robot_z = None, None, None
    
    return robot_x, robot_y, robot_z

def calculate_distance_and_orientation(robot_pos, goal_pos, robot_yaw, goal_yaw):
    """
    Calculate the distance and orientation difference between the robot and the goal.
    """
    if robot_pos is None or goal_pos is None:
        return None, None
    
    distance = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
    
    # Calculate yaw difference in degrees (0 to 360)
    yaw_diff = math.degrees(robot_yaw - goal_yaw) % 360
    if yaw_diff < 0:
        yaw_diff += 360
    
    return distance, yaw_diff


def main():
    """
    Main method of the program.
    """
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            aruco_dictionary_name))
        sys.exit(0)

    # Load the camera parameters from the saved file
    cv_file = cv2.FileStorage(
        camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
    
    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(
        aruco_dictionary_name))
    this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_dictionary_name])
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()
    
    # Start the video stream
    cap = cv2.VideoCapture(4)
    
    # Create a named window and set the window size
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 600) # Width, Height
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read() 
        
        # Get the dimensions of the frame
        frame_height, frame_width, _ = frame.shape
        frame_center = (frame_width // 2, frame_height // 2)
        
        # Draw the center point
        cv2.circle(frame, frame_center, 5, (0, 0, 255), -1) # Draw a red dot at the center
        
        # Detect ArUco markers in the video frame
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
            frame, this_aruco_dictionary, parameters=this_aruco_parameters,
            cameraMatrix=mtx, distCoeff=dst)
        
        marker_positions = {}
        robot_yaw = goal_yaw = None
        goal_position = None
        
        # Check that at least one ArUco marker was detected
        if marker_ids is not None:
            # Draw a square around detected markers in the video frame
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
            
            # Get the rotation and translation vectors
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                aruco_marker_side_length,
                mtx,
                dst)
            
            # Print the pose for the ArUco marker
            for i, marker_id in enumerate(marker_ids):
                # Store the translation (i.e. position) information
                transform_translation_x = tvecs[i][0][0]
                transform_translation_y = tvecs[i][0][1]
                transform_translation_z = tvecs[i][0][2]

                # Calculate position with respect to the center of the camera frame
                center_offset_x = transform_translation_x - frame_center[0]
                center_offset_y = transform_translation_y - frame_center[1]

                # Store the marker position
                marker_positions[marker_id[0]] = (transform_translation_x, transform_translation_y, transform_translation_z)

                # Store the rotation information
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()   
                
                # Quaternion format     
                transform_rotation_x = quat[0] 
                transform_rotation_y = quat[1] 
                transform_rotation_z = quat[2] 
                transform_rotation_w = quat[3] 
                
                # Euler angle format in radians
                roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                               transform_rotation_y, 
                                                               transform_rotation_z, 
                                                               transform_rotation_w)
                
                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)

                if marker_id[0] in [1, 2]:
                    robot_yaw = yaw_z
                elif marker_id[0] == 3:
                    goal_yaw = yaw_z
                    goal_position = (transform_translation_x, transform_translation_y, transform_translation_z)
                
                print(f"Marker ID: {marker_id[0]}")
                print("Position with respect to camera frame center:")
                print("Offset X: {:.2f} m, Offset Y: {:.2f} m".format(center_offset_x, center_offset_y))
                print("Translation: x={:.2f} m, y={:.2f} m, z={:.2f} m".format(transform_translation_x, transform_translation_y, transform_translation_z))
                print("Rotation (degrees): roll={:.2f}, pitch={:.2f}, yaw={:.2f}".format(roll_x, pitch_y, yaw_z))
                print()

                # Draw the axes on the marker
                cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)
            
        # Calculate robot position
        robot_position = calculate_robot_position(marker_positions)
        
        # If robot position and goal position are both available, calculate distance and yaw difference
        if robot_position is not None and goal_position is not None:
            distance, yaw_diff = calculate_distance_and_orientation(robot_position, goal_position, robot_yaw, goal_yaw)
            if distance is not None and yaw_diff is not None:
                print(f"Distance to goal: {distance:.2f} m, Yaw difference: {yaw_diff:.2f} degrees")
            else:
                print("Failed to calculate distance or yaw difference.")
        else:
            print("Robot position or goal position not detected.")
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # If "q" is pressed on the keyboard, 
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()
