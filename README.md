# Aruco Marker Pose Estimation

This repository contains scripts for Aruco marker pose estimation, camera calibration, and robot goal position estimation using ROS2. The workflow involves calibrating the camera, generating Aruco markers, detecting these markers, and estimating the robot's and goal's positions. The positions are then published as tf ROS topics which can be visualized in Rviz2.

## Repository Structure

```
Aruco/
├── aruco_marker_pose_estimator.py
├── camera_calibration.py
├── calibration_chessboard.yaml
├── charuco.png
├── charucoBoardCalibration.py
├── detect_aruco_marker.py
├── generate_aruco_marker.py
├── Marker/
├── pose_estimation.py
└── robot_goal_pos_extimation.py
```

## Workflow

### 1. Camera Calibration

First, you need to calibrate your camera using an Aruco marker or a chessboard pattern.

#### Using Chessboard Pattern

1. Capture images of the chessboard pattern (`charuco.png`).
2. Run the camera calibration script:

```bash
python camera_calibration.py
```

This will generate a `calibration_chessboard.yaml` file containing the camera calibration parameters.

### 2. Generate Aruco Markers

Generate specific family Aruco markers using the `generate_aruco_marker.py` script:

```bash
python generate_aruco_marker.py
```

The generated markers will be saved in the `Marker/` directory.

### 3. Detect Aruco Markers

Detect the Aruco markers in the camera frame using the `detect_aruco_marker.py` script:

```bash
python detect_aruco_marker.py
```

### 4. Pose Estimation

Estimate the pose of the detected Aruco markers using the `pose_estimation.py` script:

```bash
python pose_estimation.py
```

### 5. Robot Goal Position Estimation

Set the robot position and goal position by running the `robot_goal_pos_extimation.py` script. This script also publishes a tf ROS topic for visualization in Rviz2:

```bash
python robot_goal_pos_extimation.py
```

### Visualizing in Rviz2

To visualize the position of the robot and the goal in the camera frame using Rviz2, ensure you have the tf published by the `robot_goal_pos_extimation.py` script:

```bash
rviz2
```

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- ROS2 Foxy Fitzroy or later
- OpenCV
- Numpy

You can install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
