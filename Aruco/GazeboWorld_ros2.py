import rclpy
from rclpy.node import Node
import math
import time
import numpy as np
import cv2
import copy
import tf_transformations
import random

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent


class GazeboWorld(Node):
    def __init__(self):
        super().__init__('GazeboWorld')

        self.set_self_state1 = ModelState()
        self.set_self_state1.model_name = 'turtlebot3_waffle'
        self.set_self_state1.pose.position.x = 6.5 + np.random.uniform(-0.05, 0.05)
        self.set_self_state1.pose.position.y = -2. + np.random.uniform(-0.1, 0.1)
        self.set_self_state1.pose.position.z = 0.0
        self.set_self_state1.pose.orientation.x = 0.0
        self.set_self_state1.pose.orientation.y = 0.0
        self.set_self_state1.pose.orientation.z = 0.0
        self.set_self_state1.pose.orientation.w = 1.0
        self.set_self_state1.twist.linear.x = 0.
        self.set_self_state1.twist.linear.y = 0.
        self.set_self_state1.twist.linear.z = 0.
        self.set_self_state1.twist.angular.x = 0.
        self.set_self_state1.twist.angular.y = 0.
        self.set_self_state1.twist.angular.z = 0.
        self.set_self_state1.reference_frame = 'world'

        self.set_self_state2 = ModelState()
        self.set_self_state2.model_name = 'turtlebot3_waffle'
        self.set_self_state2.pose.position.x = -6.5 + np.random.uniform(-0.05, 0.05)
        self.set_self_state2.pose.position.y = -2. + np.random.uniform(-0.1, 0.1)
        self.set_self_state2.pose.position.z = 0.0
        self.set_self_state2.pose.orientation.x = 0.0
        self.set_self_state2.pose.orientation.y = 0.0
        self.set_self_state2.pose.orientation.z = 0.0
        self.set_self_state2.pose.orientation.w = 1.0
        self.set_self_state2.twist.linear.x = 0.
        self.set_self_state2.twist.linear.y = 0.
        self.set_self_state2.twist.linear.z = 0.
        self.set_self_state2.twist.angular.x = 0.
        self.set_self_state2.twist.angular.y = 0.
        self.set_self_state2.twist.angular.z = 0.
        self.set_self_state2.reference_frame = 'world'

        self.set_self_state3 = ModelState()
        self.set_self_state3.model_name = 'turtlebot3_waffle'
        self.set_self_state3.pose.position.x = -2.5 + np.random.uniform(-0.1, 0.1)
        self.set_self_state3.pose.position.y = -2. + np.random.uniform(-0.1, 0.1)
        self.set_self_state3.pose.position.z = 0.0
        self.set_self_state3.pose.orientation.x = 0.0
        self.set_self_state3.pose.orientation.y = 0.0
        self.set_self_state3.pose.orientation.z = 0.0
        self.set_self_state3.pose.orientation.w = 1.0
        self.set_self_state3.twist.linear.x = 0.
        self.set_self_state3.twist.linear.y = 0.
        self.set_self_state3.twist.linear.z = 0.
        self.set_self_state3.twist.angular.x = 0.
        self.set_self_state3.twist.angular.y = 0.
        self.set_self_state3.twist.angular.z = 0.
        self.set_self_state3.reference_frame = 'world'

        self.set_self_state4 = ModelState()
        self.set_self_state4.model_name = 'turtlebot3_waffle'
        self.set_self_state4.pose.position.x = 2.5 + np.random.uniform(-0.1, 0.1)
        self.set_self_state4.pose.position.y = -2. + np.random.uniform(-0.1, 0.1)
        self.set_self_state4.pose.position.z = 0.0
        self.set_self_state4.pose.orientation.x = 0.0
        self.set_self_state4.pose.orientation.y = 0.0
        self.set_self_state4.pose.orientation.z = 0.0
        self.set_self_state4.pose.orientation.w = 1.0
        self.set_self_state4.twist.linear.x = 0.
        self.set_self_state4.twist.linear.y = 0.
        self.set_self_state4.twist.linear.z = 0.
        self.set_self_state4.twist.angular.x = 0.
        self.set_self_state4.twist.angular.y = 0.
        self.set_self_state4.twist.angular.z = 0.
        self.set_self_state4.reference_frame = 'world'

        self.set_self_state5 = ModelState()
        self.set_self_state5.model_name = 'turtlebot3_waffle'
        self.set_self_state5.pose.position.x = 0. + np.random.uniform(-0.1, 0.1)
        self.set_self_state5.pose.position.y = 7. + np.random.uniform(-0.1, 0.1)
        self.set_self_state5.pose.position.z = 0.0
        self.set_self_state5.pose.orientation.x = 0.0
        self.set_self_state5.pose.orientation.y = 0.0
        self.set_self_state5.pose.orientation.z = 0.0
        self.set_self_state5.pose.orientation.w = 1.0
        self.set_self_state5.twist.linear.x = 0.
        self.set_self_state5.twist.linear.y = 0.
        self.set_self_state5.twist.linear.z = 0.
        self.set_self_state5.twist.angular.x = 0.
        self.set_self_state5.twist.angular.y = 0.
        self.set_self_state5.twist.angular.z = 0.
        self.set_self_state5.reference_frame = 'world'

        self.depth_image_size = [160, 128]
        self.rgb_image_size = [304, 228]
        self.bridge = CvBridge()

        self.object_state = [0, 0, 0, 0]
        self.object_name = []

        self.action_table = [np.pi / 6, np.pi / 12, 0., -np.pi / 12, -np.pi / 6]

        self.self_speed = [.2, 0.0]
        self.default_states = None

        self.start_time = time.time()
        self.max_steps = 10000

        self.depth_image = None
        self.bump = False

        self.cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.set_state = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.resized_depth_img = self.create_publisher(Image, 'camera/depth/image_resized', 10)
        self.resized_rgb_img = self.create_publisher(Image, 'camera/rgb/image_resized', 10)

        self.object_state_sub = self.create_subscription(ModelStates, 'gazebo/model_states', self.ModelStateCallBack, 10)
        self.rgb_image_sub = self.create_subscription(Image, 'camera/rgb/image_raw', self.RGBImageCallBack, 10)
        self.depth_image_sub = self.create_subscription(Image, 'camera/depth/image_raw', self.DepthImageCallBack, 10)
        self.laser_sub = self.create_subscription(LaserScan, 'scan', self.LaserScanCallBack, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.OdometryCallBack, 10)
        self.bumper_sub = self.create_subscription(BumperEvent, 'mobile_base/events/bumper', self.BumperCallBack, 10)

        self.create_timer(2.0, self.timer_callback)

        self.add_on_shutdown(self.shutdown)

    def ModelStateCallBack(self, data):
        idx = data.name.index('turtlebot3_waffle')
        quaternion = (data.pose[idx].orientation.x,
                      data.pose[idx].orientation.y,
                      data.pose[idx].orientation.z,
                      data.pose[idx].orientation.w)
        euler = tf_transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        self.self_state = [data.pose[idx].position.x,
                           data.pose[idx].position.y,
                           yaw,
                           data.twist[idx].linear.x,
                           data.twist[idx].linear.y,
                           data.twist[idx].angular.z]
        for lp in range(len(self.object_name)):
            idx = data.name.index(self.object_name[lp])
            quaternion = (data.pose[idx].orientation.x,
                          data.pose[idx].orientation.y,
                          data.pose[idx].orientation.z,
                          data.pose[idx].orientation.w)
            euler = tf_transformations.euler_from_quaternion(quaternion)
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            self.object_state[lp] = [data.pose[idx].position.x,
                                     data.pose[idx].position.y,
                                     yaw]
        if self.default_states is None:
            self.default_states = copy.deepcopy(data)

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def LaserScanCallBack(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)

    def OdometryCallBack(self, odometry):
        self.self_position_x = odometry.pose.pose.position.x
        self.self_position_y = odometry.pose.pose.position.y
        self.quaternion = (odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y,
                           odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w)
        self.self_linear_x_speed = odometry.twist.twist.linear.x
        self.self_linear_y_speed = odometry.twist.twist.linear.y
        self.self_rotation_z_speed = odometry.twist.twist.angular.z

    def BumperCallBack(self, bumper_data):
        if bumper_data.state == BumperEvent.PRESSED:
            self.bump = True
        else:
            self.bump = False

    def GetDepthImageObservation(self):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
        except Exception as e:
            raise e
        try:
            cv_rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        cv_img = np.array(cv_img, dtype=np.float32)
        dim = (self.depth_image_size[0], self.depth_image_size[1])
        cv_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)

        cv_img[np.isnan(cv_img)] = 0.
        cv_img[cv_img < 0.4] = 0.

        gauss = np.random.normal(0., 0.15, dim)
        gauss = gauss.reshape(dim[1], dim[0])
        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img = cv_img + gauss
        cv_img[cv_img < 0.4] = 0.

        cv_img = np.array(cv_img, dtype=np.float32)

        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)
        return (cv_img)

    def GetRGBImageObservation(self):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        self.resized_rgb_img.publish(resized_img)
        return (cv_resized_img)

    def PublishDepthPrediction(self, depth_img):
        cv_img = np.array(depth_img, dtype=np.float32)
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)

    def GetLaserObservation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isinf(scan)] = 30.
        return scan

    def GetSelfState(self):
        return self.self_state

    def GetSelfLinearXSpeed(self):
        return self.self_linear_x_speed

    def GetSelfOdomeInfo(self):
        euler = tf_transformations.euler_from_quaternion(self.quaternion)
        Eular = euler[2]
        v = np.sqrt(self.self_linear_x_speed ** 2 + self.self_linear_y_speed ** 2)
        return [v, self.self_rotation_z_speed, Eular]

    def GetGoalInfo(self):
        euler = tf_transformations.euler_from_quaternion(self.quaternion)
        yaw = euler[2]
        R2G = np.array(np.array(self.goal) - np.array([self.self_position_x, self.self_position_y]))
        distance = np.sqrt(R2G[0] ** 2 + R2G[1] ** 2)
        if yaw < 0:
            yaw = yaw + 2 * np.pi
        rob_ori = np.array([np.cos(yaw), np.sin(yaw)])
        angle = np.arccos(R2G.dot(rob_ori) / np.sqrt((R2G.dot(R2G)) * np.sqrt(rob_ori.dot(rob_ori))))

        if rob_ori[0] > 0 and (rob_ori[1] / rob_ori[0]) * R2G[0] > R2G[1]:
            angle = -angle
        elif rob_ori[0] < 0 and (rob_ori[1] / rob_ori[0]) * R2G[0] < R2G[1]:
            angle = -angle
        elif rob_ori[0] == 0:
            if rob_ori[1] > 0 and R2G[0] > 0:
                angle = -angle
            elif rob_ori[1] < 0 and R2G[0] < 0:
                angle = -angle

        goal = np.array([distance, angle])
        self.get_logger().info(f"goal_position, current_position, distance, angle: {self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.self_position_x:.2f}, {self.self_position_y:.2f}, {goal[0]:.2f}, {goal[1]:.2f}")
        return goal

    def GetTargetState(self, name):
        return self.object_state[self.TargetName.index(name)]

    def GetSelfSpeed(self):
        return np.array(self.self_speed)

    def GetBump(self):
        return self.bump

    def SetObjectPose(self, name='mobile_base', random_flag=False):
        if name == 'mobile_base':
            rand = random.random()
            if rand < 0.2:
                object_state = copy.deepcopy(self.set_self_state1)
                quaternion = tf_transformations.quaternion_from_euler(0., 0., np.pi / 2 + np.random.uniform(-np.pi / 6, np.pi / 6))
                self.goal = (np.random.uniform(-1, 1), 11 + np.random.uniform(-0.4, 0.4))

            elif rand < 0.4:
                object_state = copy.deepcopy(self.set_self_state2)
                quaternion = tf_transformations.quaternion_from_euler(0., 0., np.pi / 2 + np.random.uniform(-np.pi / 6, np.pi / 6))
                self.goal = (np.random.uniform(-1, 1), 11 + np.random.uniform(-0.4, 0.4))

            elif rand < 0.6:
                object_state = copy.deepcopy(self.set_self_state3)
                quaternion = tf_transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi / 4, 3 * np.pi / 4))
                if random.random() < 0.5:
                    self.goal = (-1.5 + np.random.uniform(-0.5, 0.5), 6.5 + np.random.uniform(-0.5, 0.5))
                else:
                    self.goal = (1.5 + np.random.uniform(-0.5, 0.5), 6.5 + np.random.uniform(-0.5, 0.5))

            elif rand < 0.8:
                random_angle = np.random.choice([np.random.uniform(-np.pi, -3 * np.pi / 4),
                                                 np.random.uniform(np.pi / 4, np.pi / 2),
                                                 np.random.uniform(np.pi / 2, 3 * np.pi / 4),
                                                 np.random.uniform(3 * np.pi / 4, np.pi)])
                object_state = copy.deepcopy(self.set_self_state4)
                quaternion = tf_transformations.quaternion_from_euler(0., 0., random_angle)
                if random.random() < 0.5:
                    self.goal = (-1.5 + np.random.uniform(-0.5, 0.5), 6.5 + np.random.uniform(-0.5, 0.5))
                else:
                    self.goal = (1.5 + np.random.uniform(-0.5, 0.5), 6.5 + np.random.uniform(-0.5, 0.5))

            else:
                object_state = copy.deepcopy(self.set_self_state5)
                quaternion = tf_transformations.quaternion_from_euler(0., 0., np.random.uniform(-5 * np.pi / 6, -np.pi / 6))
                if random.random() < 0.5:
                    self.goal = (-2.5 + np.random.uniform(-0.1, 0.1), -2 + np.random.uniform(-0.1, 0.1))
                else:
                    self.goal = (2.5 + np.random.uniform(-0.1, 0.1), -2 + np.random.uniform(-0.1, 0.1))

            R2G = np.array(np.array(self.goal) - np.array(
                [object_state.pose.position.x, object_state.pose.position.y]))
            self.ini_dis = np.sqrt(R2G[0] ** 2 + R2G[1] ** 2)
            object_state.pose.orientation.x = quaternion[0]
            object_state.pose.orientation.y = quaternion[1]
            object_state.pose.orientation.z = quaternion[2]
            object_state.pose.orientation.w = quaternion[3]
        else:
            object_state = self.States2State(self.default_states, name)

        self.set_state.publish(object_state)

    def States2State(self, states, name):
        to_state = ModelState()
        from_states = copy.deepcopy(states)
        idx = from_states.name.index(name)
        to_state.model_name = name
        to_state.pose = from_states.pose[idx]
        to_state.twist = from_states.twist[idx]
        to_state.reference_frame = 'world'
        return to_state

    def ResetWorld(self):
        self.total_evaluation = 0
        self.SetObjectPose()
        self.self_speed = [.4, 0.0]
        self.step_target = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        time.sleep(0.5)

    def Control(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = 0.25
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = self.action_table[action]
        self.cmd_vel.publish(move_cmd)

    def shutdown(self):
        self.get_logger().info("Stop Moving")
        self.cmd_vel.publish(Twist())
        time.sleep(1)

    def GetRewardAndTerminate(self, t):
        terminate = False
        reset = False
        [v, theta, Eular] = self.GetSelfOdomeInfo()
        laser = self.GetLaserObservation()
        Laser = []
        goal = self.GetGoalInfo()
        for i in range(-90, 90):
            Laser = np.append(Laser, laser[i])
        Distance = np.amin(Laser)
        Angle = np.abs(np.argmin(Laser) - 90)
        reward_ob = 0
        reward_reach_g = 0

        if 0.6 < Distance < 1.2:
            reward_ob = (- 2.4 + 2 * Distance) * (180 - Angle) / 135
        if self.GetBump() or Distance < 0.6:
            reward_ob = (-8.0) * ((((180. - Angle) / 135. - 1.) * 3. / 8.) + 1.)
            terminate = True
            reset = True

        reward_goal = (-np.abs(goal[1]) + np.pi / 3) / 10
        if Distance < 1.2:
            reward_goal = reward_goal * (Distance - 0.6) / 0.6

        if goal[0] < 0.2:
            reward_reach_g = 5. * np.cos(goal[1]) + 1
            terminate = True
            reset = True
            self.get_logger().info("Reach the goal!!!")

        if t > 500:
            reset = True
            self.get_logger().info("Didn't reach the goal")

        reward = reward_reach_g + reward_goal + reward_ob

        self.total_evaluation = self.total_evaluation + reward
        total_evaluation = self.total_evaluation

        return reward, terminate, reset, total_evaluation, goal

    def timer_callback(self):
        pass  # You can define any periodic tasks here


def main(args=None):
    rclpy.init(args=args)
    gazebo_world = GazeboWorld()

    try:
        rclpy.spin(gazebo_world)
    except KeyboardInterrupt:
        pass

    gazebo_world.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

