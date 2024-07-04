#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import numpy as np
import yaml

def saveCameraParams(filename, imageSize, cameraMatrix, distCoeffs, totalAvgErr):
    print(cameraMatrix)
    
    calibration = {'camera_matrix': cameraMatrix.tolist(), 'distortion_coefficients': distCoeffs.tolist()}

    calibrationData = dict(
        image_width=imageSize[0],
        image_height=imageSize[1],
        camera_matrix=dict(
            rows=cameraMatrix.shape[0],
            cols=cameraMatrix.shape[1],
            dt='d',
            data=cameraMatrix.tolist(),
        ),
        distortion_coefficients=dict(
            rows=distCoeffs.shape[0],
            cols=distCoeffs.shape[1],
            dt='d',
            data=distCoeffs.tolist(),
        ),
        avg_reprojection_error=totalAvgErr,
    )

    with open(filename, 'w') as outfile:
        yaml.dump(calibrationData, outfile)

# Parameters
output_filename = "calibration.yml"
square_size = 0.035  # Size of squares in meters
marker_size = 0.0175  # Size of ArUco markers in meters
dictionary_id = aruco.DICT_4X4_50
num_squares_x = 10  # Number of squares in X direction
num_squares_y = 6   # Number of squares in Y direction

# Create the charuco board
dictionary = aruco.getPredefinedDictionary(dictionary_id)
board = aruco.CharucoBoard_create(num_squares_x, num_squares_y, square_size, marker_size, dictionary)

# Initialize lists to store corners and ids from all images
allCorners = []
allIds = []

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'c' to capture images for calibration. Press 'q' to quit and start calibration.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markerCorners, markerIds, rejectedImgPoints = aruco.detectMarkers(gray, dictionary)

    if markerIds is not None:
        aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(markerCorners, markerIds, gray, board)
        if charucoCorners is not None and charucoIds is not None:
            aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if markerIds is not None and len(markerIds) > 0:
            print("Captured frame for calibration.")
            allCorners.append(charucoCorners)
            allIds.append(charucoIds)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
print("Calibrating camera...")

if len(allCorners) > 0:
    imsize = gray.shape
    try:
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
        print("Calibration successful.")
        print("Reprojection error:", ret)
        saveCameraParams(output_filename, imsize, cameraMatrix, distCoeffs, ret)
    except Exception as e:
        print(f"Calibration failed: {e}")
else:
    print("No corners were found for calibration.")
