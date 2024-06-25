#!/usr/bin/env python

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
    cap = cv2.VideoCapture(-2)
    
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
                
                print(f"Marker ID: {marker_id[0]}")
                print("Position with respect to camera frame center:")
                print("Offset X: {:.2f} m, Offset Y: {:.2f} m".format(center_offset_x, center_offset_y))
                print("Translation: x={:.2f} m, y={:.2f} m, z={:.2f} m".format(transform_translation_x, transform_translation_y, transform_translation_z))
                print("Rotation (degrees): roll={:.2f}, pitch={:.2f}, yaw={:.2f}".format(roll_x, pitch_y, yaw_z))
                print()

                # Draw the axes on the marker
                cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)
            
            # Calculate distances between each pair of markers
            marker_ids_list = list(marker_positions.keys())
            for j in range(len(marker_ids_list)):
                for k in range(j + 1, len(marker_ids_list)):
                    id1 = marker_ids_list[j]
                    id2 = marker_ids_list[k]
                    pos1 = marker_positions[id1]
                    pos2 = marker_positions[id2]
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    print(f"Distance between Marker {id1} and Marker {id2}: {distance:.2f} m")
            
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
