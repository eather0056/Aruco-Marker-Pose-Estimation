# BlueROV Navigator Setup Instructions

## Overview
The new BlueROV systems now utilize a Navigator board connected directly to a Raspberry Pi 4 (RPI4) running a Linux OS with the BlueOS layer from BlueRobotics. This replaces the older setup which used a Pixhawk board linked to a Raspberry Pi 3 (RPI3) via USB cable, running a Linux/ROS1 system onboard. The MAVROS node will now run on your laptop instead of onboard, communicating with the MAVlink endpoint client on the BlueOS. Below are the detailed setup instructions for both the laptop and the embedded system.

## Laptop Setup

### 1. Install Ubuntu 22.04 LTS
Make sure your laptop is running Ubuntu 22.04 LTS.

### 2. Install ROS 2
Install either ROS 2 Humble or ROS 2 Iron. This guide uses ROS 2 Iron.
```sh
sudo apt update
sudo apt install ros-iron-desktop
```

### 3. Install MAVROS and MAVlink
Install the required ROS 2 packages for MAVROS and MAVlink.
```sh
sudo apt-get install ros-iron-mavros ros-iron-mavlink ros-iron-mavros-extras ros-iron-mavros-msgs
```

### 4. Install Python3 Tools
Ensure Python3 tools are installed.
```sh
sudo apt install -y python3-vcstool python3-rosinstall-generator python3-osrf-pycommon
```

### 5. Install GeographicLib Datasets
Download and install the GeographicLib datasets.
```sh
wget https://raw.githubusercontent.com/mavlink/mavros/ros2/mavros/scripts/install_geographiclib_datasets.sh
chmod u+x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```

### 6. Install 3D Transformations Tools
Install additional packages for 3D transformations.
```sh
sudo apt-get install ros-iron-tf-transformations
sudo pip3 install transform3d
```

### 7. Create Workspace
Create a ROS 2 workspace if not already done.
```sh
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 8. Install ROS 2 Package `autonomous_rov`
Extract the provided `autonomous_rov` package into the `src` folder.
```sh
tar xzvf path/to/pack-autonomous-rov.tgz -C ~/ros2_ws/src
```

### 9. Compile the Workspace
Build the workspace.
```sh
cd ~/ros2_ws
colcon build
```

### 10. Source the Workspace
Source the workspace setup file.
```sh
source install/setup.bash
```

### 11. Setup Network Connection
Create a network connection with the BlueOS box.
- Connect the USB cable of the BlueOS box to your laptop.
- Configure the connection with IP: `192.168.2.1` and Netmask: `255.255.255.0`.

## Embedded Board Setup

### 1. Download ArduSub Firmware
Download the updated ArduSub firmware.
```plaintext
https://filesender.renater.fr/?s=download&token=6e23a41e-86a7-448d-9902-3243915695ab
```

### 2. Connect to BlueOS
- Open a web browser and go to `http://192.168.2.2`.
- Follow the instructions to upload the firmware.

### 3. Reboot
Reboot the system.

### 4. Configure Frame in BlueOS
- Connect to BlueOS via `http://192.168.2.2`.
- Change the frame configuration to a custom one.
- Restart the ArduSub ardupilot.

## Running MAVROS and Launch Files on Laptop

### 1. Connect the Gamepad
Ensure the gamepad is connected to your laptop. Source the setup.bash
```sh
source /home/eth/ros2_ws/src/install/setup.bash
```
### 2. Run the Launch File
Run the provided launch file to start the MAVROS, joy, and listenMIR nodes.
```sh
ros2 launch autonomous_rov run_all.launch
```
If the gamepad is not detected, modify the `joy_dev` parameter inside the launch file.

### 3. Modify and Build ListenerMIR Node
Implement your algorithm in `listenerMIR.py`.
- After making changes, rebuild the workspace.
```sh
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 4. Display Camera Video
Use the provided launch file to display camera video.
```sh
ros2 launch autonomous_rov video.launch.py
```
Check the list of your laptop camera.
```sh
sudo apt-get install v4l-utils
v4l2-ctl --list-devices
```

### 5. Run QGroundControl (QGC) Interface
You can run QGroundControl to perform calibration steps without changes inside the RPI. Note that QGC may crash sometimes after calibration and may not be fully compatible with all commands.

---

## Notes
- Ensure that each time you modify a launch file or a Python file, you rebuild and source the workspace.
- This guide assumes a working knowledge of ROS 2, Linux, and networking. Adjust IP addresses and parameters as needed for your specific setup.
