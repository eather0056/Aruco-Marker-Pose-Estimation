#!/usr/bin/env python3

import numpy as np
import cv2
import cv2.aruco as aruco
import argparse
import sys
import yaml
import numpy as np

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

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", help="output calibration filename", default="calibration.yml")
parser.add_argument("-s", "--size", help="size of squares in meters", type=float, default="0.035")
parser.add_argument('imgs', nargs='+', help='list of images for calibration')
args = parser.parse_args()

sqWidth = 12  # number of squares width
sqHeight = 8  # number of squares height
allCorners = []  # all Charuco Corners
allIds = []  # all Charuco Ids
decimator = 0

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(sqWidth, sqHeight, 0.035, 0.0175, dictionary)

for f in args.imgs:
    print("reading %s" % f)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not read file {f}")
        continue  # Skip to the next file
    
    try:
        [markerCorners, markerIds, rejectedImgPoints] = cv2.aruco.detectMarkers(img, dictionary)

        if len(markerCorners) > 0:
            [ret, charucoCorners, charucoIds] = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, img, board)
            if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3 and decimator % 3 == 0:
                allCorners.append(charucoCorners)
                allIds.append(charucoIds)

            cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
            cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)

        smallimg = cv2.resize(img, (1024, 768))
        cv2.imshow("frame", smallimg)
        cv2.waitKey(0)  # any key
        decimator += 1

    except Exception as e:
        print(f"Error processing file {f}: {e}")

imsize = img.shape if img is not None else (0, 0)
print(imsize)

# try Calibration
try:
    [ret, cameraMatrix, distCoeffs, rvecs, tvecs] = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
    print("Rep Error:", ret)
    saveCameraParams(args.file, imsize, cameraMatrix, distCoeffs, ret)

except ValueError as e:
    print(e)
except NameError as e:
    print(e)
except AttributeError as e:
    print(e)
except:
    print("calibrateCameraCharuco fail:", sys.exc_info()[0])

print("Press any key on window to exit")
cv2.waitKey(0)  # any key
cv2.destroyAllWindows()
