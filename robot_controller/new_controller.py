import rclpy
import rclpy.exceptions
import rclpy.executors
from rclpy.node import Node

import threading
import time

import numpy as np

from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation as R

import rclpy.time

from .robotiq_gripper import RobotiqGripper as robotiq_gripper

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .jacobian.jacobian_src import get_jacobian, get_ee_position

from .breathing import breathing_src as breathing

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from enum import Enum

class RobotState(Enum):
    START = 0
    PICKING = 1
    PLACING = 2
    BREATHING_GAZING = 3
    END = 4

data = []
data_c = []

class GripperController:
    def __init__(self, host, port):
        self.host = host # The robot's computer address
        self.port = port # The gripper's port number
        self.gripper = None
        self._init_gripper()

    def _init_gripper(self):
        print("Creating gripper...")
        self.gripper = robotiq_gripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.host, self.port)
        print("Activating gripper...")
        self.gripper.activate(auto_calibrate=False)
        
        self.close()

    def open(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(0, speed, force)
        # self.pub.publish(String("0.0"))

    def close(self, speed=255, force=255):
        self.gripper.move_and_wait_for_pos(255, speed, force)
        # self.pub.publish(String("0.7"))

    def close_async(self, speed=255, force=255):
        self.gripper.move(255, speed, force)
        
    def open_async(self, speed=255, force=255):
        self.gripper.move(0, speed, force)

    def move_gripper(self, pos, speed=255, force=255):
        pos = 0 if pos < 0 else (255 if pos > 255 else pos)
        self.gripper.move_and_wait_for_pos(pos, speed, force)
        
    def current_pos(self):        
        return self.gripper.get_current_position()
    
    def is_holding(self):
        prev_pos = self.gripper.get_current_position()
        if prev_pos < 50:
            return False
        #else:
        #    print("WHY ARE WE HERE, JUST TO SUFFER?!")
        self.gripper.move_and_wait_for_pos(prev_pos+1,255,255)
        if (abs(self.gripper.get_current_position()-prev_pos)==0) and (self.gripper.get_current_position() < 100):
            return True
        else:
            return False

def make_parabola(start_pos, mid_pos, end_pos, amt_points=100):
    
    x_axis = (end_pos - start_pos) / np.linalg.norm(end_pos - start_pos)
    z_axis = np.cross(x_axis, mid_pos - start_pos)
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm
    orig_axis = start_pos #not an axis, dont know why i called it that - sometimes somethings just happen and all you can do is accept and move on
    
    x_axis = np.concatenate((x_axis, [0]))
    y_axis = np.concatenate((y_axis, [0]))
    z_axis = np.concatenate((z_axis, [0]))
    orig_axis = np.concatenate((orig_axis, [1]))
    
    transform_matrix = np.array([x_axis, y_axis, z_axis, orig_axis])
    transform_matrix = np.transpose(transform_matrix)
    
    print("Transform Matrix: ", transform_matrix)
    
    # -------------------------------
    
    vec_ie = end_pos - start_pos
    vec_im = mid_pos - start_pos
    
    norm_ie = np.linalg.norm(vec_ie)
    norm_im = np.linalg.norm(vec_im)
    
    cosalfa = np.dot(vec_ie, vec_im) / (norm_ie * norm_im)
    sinalfa = np.sqrt(1 - cosalfa**2)
        
    vec_xm = cosalfa*norm_im
    vec_ym = sinalfa*norm_im
    
    a = vec_ym/(vec_xm*vec_xm - vec_xm*norm_ie)
    b = -a*norm_ie
        
    steps = np.linspace(0,1,num=amt_points,endpoint=False)
    sample_points_x = np.interp(steps,[0,1],[0,norm_ie])    
    
    print("Sample Points: ", sample_points_x)
    sample_points_y = []
    
    for i in range(len(sample_points_x)):        
        sample_points_y.append(a*sample_points_x[i]**2 + b*sample_points_x[i])
        
    traj_points = []
        
    for i in range(len(sample_points_x)):        
        traj_points.append((transform_matrix @ np.array([sample_points_x[i], sample_points_y[i], 0, 1]))[0:3])
        print("Traj: ", traj_points[-1])
    
    return traj_points

def three_point_cubic_spline(start_pos, mid_pos, end_pos):
    p0 = start_pos
    p1 = end_pos    
    r0 = (mid_pos - start_pos) / np.linalg.norm(mid_pos - start_pos)
    r1 = (end_pos - mid_pos) / np.linalg.norm(end_pos - mid_pos)
    
    p0x, p0y, p0z = p0[0], p0[1], p0[2]
    p1x, p1y, p1z = p1[0], p1[1], p1[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    r1x, r1y, r1z = r1[0], r1[1], r1[2]
    
    x_constraints = np.array([p0x, p1x, r0x, r1x])
    y_constraints = np.array([p0y, p1y, r0y, r1y])
    z_constraints = np.array([p0z, p1z, r0z, r1z])
    
    cubic_spline_inverse_matrix = np.array([[2, -2, 1, 1],
                                            [-3, 3, -2, -1],
                                            [0, 0, 1, 0],
                                            [1, 0, 0, 0]])

    
    x_coefficients = cubic_spline_inverse_matrix @ x_constraints
    y_coefficients = cubic_spline_inverse_matrix @ y_constraints
    z_coefficients = cubic_spline_inverse_matrix @ z_constraints
    
    return x_coefficients, y_coefficients, z_coefficients

class NewController(Node):

    def init_joint_states(self):
        # Get joint states
        self.joint_states_global = {}
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        while not self.joint_states_global:
            #self.get_logger().info('Waiting for joint states...')
            rclpy.spin_once(self)
        self.get_logger().info('Joint states received.')
        
    def init_velocity_controller(self):
        # Publish velocity commands
        self.velocityControllerTopic = "forward_velocity_controller/commands" # All movements should be done w.r.t. "base", not "base_link" or "world"
        self.velocityControllerPub = self.create_publisher(Float64MultiArray, self.velocityControllerTopic, 10)
        
        while not self.velocityControllerPub.get_subscription_count():
            #self.get_logger().info('Waiting for velocity controller to connect...')
            rclpy.spin_once(self)
        self.get_logger().info('Velocity controller connected.')

    def init_tf(self):
        # Get tf tree
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        while not self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()):
            #self.get_logger().info('Waiting for tf tree...')
            rclpy.spin_once(self)
        self.get_logger().info('TF tree received.')

    def init_gripper(self):
        # Initialize gripper
        self.robot_ip = "192.168.1.102"
        self.gripper_port = 63352
        self.gripper = GripperController(self.robot_ip, self.gripper_port)
        
        self.gripper.open()
        self.get_logger().info('Gripper opened.')
        self.gripper.close()
        self.get_logger().info('Gripper closed, gripper ready to go.')

    def init_breather(self):
        # Set breathing parameters 
        self.breathe_dict = {}
        self.breathe_dict["control_rate"] = self.control_rate

        # Set breathe vector direction in base frame
        breathe_vec = np.array([0., 0., 1.])  # named as r in the paper
        breathe_vec = breathe_vec / np.linalg.norm(breathe_vec) if np.all(breathe_vec > 0) else breathe_vec
        self.breathe_dict["breathe_vec"] = breathe_vec

        # Human breathing data based breathing velocity profile function f(·)
        human_data_path = "/home/kovan/USTA/src/robot_controller/robot_controller/breathing/breathe_data.npy"
        f = np.load(human_data_path)               
        self.breathe_dict["f"] = f

        # FAST: period = 2, amplitude = 1.2
        # DEFAULT: period = 4, amplitude = 1.0

        # session_speed = exp_speed.Adaptive # deal with later
        
        self.breathe_period, amplitude = 2, 1.4 #get_session_parameters(session_speed)
        freq = 1.0 / self.breathe_period  # Named as beta in the paper
        self.breathe_dict["freq"] = freq
        self.breathe_dict["amplitude"] = amplitude

        self.num_of_total_joints = 6  # UR5e has 6 joints
        # Use at least 3 joints for breathing, below that, it is not effective
        self.num_of_breathing_joints = 3  # Number of body joints starting from base towards to end effector
        self.breathe_dict["num_of_joints"] = self.num_of_breathing_joints

        self.breathe_dict["compensation"] = False
        self.breathe_dict["filter"] = True
        self.breathe_controller = breathing.Breather(self.breathe_dict)

    def __init__(self):
        super().__init__('new_controller')
        # Sanity check at the start of node (The phrases are very unprofessionally taken from my favorite anime, Serial Experiments Lain)
        self.get_logger().info('Present day,')
        
        # Initialize control rate
        self.control_rate = 500  # Hz
        self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        
        self.base = "base"
        self.eef = "wrist_3_link"
        self.world = "world"
        
        self.init_joint_states()
        
        self.init_velocity_controller()
        
        self.init_tf()
        
        self.init_gripper()
        
        self.init_breather()
        self.use_helmet = True
        
        self.vel_comm_ind = 0
        
        self.prev_velocities = np.zeros(6)
        
        self.home_pos = np.array([-0.8, -1.73, -2.1,  0.73,  1.52,  3.16])
        self.gripper_length = 0.17
        
        self.traj_publisher = self.create_publisher(MarkerArray, "trajectory", 10)
        
        self.state = RobotState.START
        self.speed = 0.05
                                        
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('present time.')  
    
    def get_gaze_velocities(self, breathing_velocities):
        
        tf_name = "eye" if self.use_helmet else "exp/head"
                    
        gaze_multiplier = 3
        
        cw1 = 0
        cw2 = 0                
        
        jacobian = self.get_jacobian_matrix()
        
        vels = np.concatenate((breathing_velocities, self.joint_states_global["vels"][self.num_of_breathing_joints:]))
        instant_velocities = jacobian @ vels
        delta = instant_velocities[:3] * self.breathe_period
        #delta *= 0 # Comment this line to enable gazing
        
        gazing_velocities = np.zeros(self.num_of_total_joints - self.num_of_breathing_joints)
        try:
            
            base_to_w1 = self.tfBuffer.lookup_transform("wrist_1_link", "base", rclpy.time.Time(seconds=0))
            delta_in_w1 = R.from_quat([base_to_w1.transform.rotation.x,
                                       base_to_w1.transform.rotation.y,
                                       base_to_w1.transform.rotation.z,
                                       base_to_w1.transform.rotation.w]).apply(delta)
            
            base_to_w2 = self.tfBuffer.lookup_transform("wrist_2_link", "base", rclpy.time.Time(seconds=0))
            delta_in_w2 = R.from_quat([base_to_w2.transform.rotation.x,
                                        base_to_w2.transform.rotation.y,
                                        base_to_w2.transform.rotation.z,
                                        base_to_w2.transform.rotation.w]).apply(delta)
                        
            head_in_w1 = self.tfBuffer.lookup_transform("wrist_1_link", tf_name, rclpy.time.Time(seconds=0))
            head_in_w1_pos = np.array([head_in_w1.transform.translation.x, head_in_w1.transform.translation.y-delta_in_w1[1], 0])
            head_in_w1_norm = np.linalg.norm(head_in_w1_pos)
            cw1 = np.arccos(0.0997 / head_in_w1_norm) - np.arccos(np.dot([0,-1,0], head_in_w1_pos/head_in_w1_norm))
            cw1 = cw1*(-1) if head_in_w1_pos[0] > 0 else cw1
            gazing_velocities[0] = cw1*gaze_multiplier
            
            head_in_w2 = self.tfBuffer.lookup_transform("wrist_2_link", tf_name, rclpy.time.Time(seconds=0))
            head_in_w2_pos = np.array([head_in_w2.transform.translation.x, head_in_w2.transform.translation.y-delta_in_w2[1], 0])
            head_in_w2_norm = np.linalg.norm(head_in_w2_pos)
            cw2 = np.arccos(np.dot([0,1,0], head_in_w2_pos/head_in_w2_norm))
            cw2 = cw2*(-1) if head_in_w2_pos[0] > 0 else cw2
            gazing_velocities[1] = cw2*gaze_multiplier
        except TransformException as e:
            gazing_velocities = np.zeros(self.num_of_total_joints - self.num_of_breathing_joints)
            print("Error in getting head position: ", e)
        finally:
            return gazing_velocities
        
    def breathe_and_gaze(self, do_breathing=True, do_gazing=True):   
        
        print("Starting breathe and gaze.") 
        
        self.gripper.close_async()
                            
        self.breathe_controller.reset()   
        
        if do_gazing:
            print("gaze target at: ", self.tfBuffer.lookup_transform("world", "eye", rclpy.time.Time()).transform.translation) 
            
        while rclpy.ok():      
                        
            breathing_velocities = np.zeros(self.num_of_breathing_joints)
            if do_breathing:
                breathing_velocities = self.breathe_controller.step(self.joint_states_global["pos"], self.joint_states_global["vels"], get_jacobian)
            
            gazing_velocities = np.zeros(self.num_of_total_joints - self.num_of_breathing_joints)
            if do_gazing:                
                gazing_velocities = self.get_gaze_velocities(breathing_velocities)                
                    
            # Publish joint vels to robot
            velocity_command = np.concatenate((breathing_velocities, gazing_velocities))
                                 
            self.publishVelocityCommand(velocity_command)
                                                
            self.ros_rate.sleep()
                        
        print("Exiting breathe and gaze.")
            
        self.stop_movement() 

    def get_current_coordinate(self):
        current_coordinate = np.zeros(3)
        
        try:
            current_coordinate = get_ee_position(self.joint_states_global["pos"])
        except:
            print("Error in getting current coordinate.")
        return current_coordinate
        
        
        """current_coordinate = np.zeros(3)
        try:
            transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
            current_coordinate[0] = transformation.transform.translation.x
            current_coordinate[1] = transformation.transform.translation.y
            current_coordinate[2] = transformation.transform.translation.z 
        except:
            print("Error in getting current coordinate.")
        return current_coordinate"""

    def get_current_orientation(self):
        current_orientation = np.zeros(3)
        try:
            transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
            rotvec = R.from_quat([transformation.transform.rotation.x,
                                    transformation.transform.rotation.y,
                                    transformation.transform.rotation.z,
                                    transformation.transform.rotation.w]).as_rotvec()
            current_orientation = rotvec / np.linalg.norm(rotvec)
        except:
            print("Error in getting current orientation.")
        return current_orientation

    def go_to_pose_in_base(self, desired_coordinate, desired_orientation=None, speed=0.05):       
        
        current_coordinate = self.get_current_coordinate()
        
        max_error = 0.001
        
        dist = np.linalg.norm(desired_coordinate - current_coordinate)
        slowdown_dist = dist*0.2
        
        if np.linalg.norm(desired_coordinate - current_coordinate) < max_error:
            self.stop_movement()
            print("Already at the desired position.")
            return
                                 
        print("Current position: ", current_coordinate)   
        print("Going to point: ", desired_coordinate)
        
        est_time = np.linalg.norm(desired_coordinate - current_coordinate) / speed
        
        if desired_orientation is not None:
            rot_speed = np.linalg.norm(desired_orientation)/est_time            
            desired_rot_speed_in_base = (desired_orientation / np.linalg.norm(desired_orientation)) * rot_speed 
            desired_orientation = desired_orientation / np.linalg.norm(desired_orientation)          
            print("Desired Rot Speed: ", desired_rot_speed_in_base)
        else:
            print("No desired orientation, keeping current orientation.")
            desired_rot_speed_in_base = np.zeros(3)
                    
        start_time = time.time()                
        min_speed = 0.05
        min_dist = 0.1
        curr_speed = speed if dist > min_dist else 0.1
        error = 99999
                                    
        while rclpy.ok() and error > max_error:
                        
            if time.time() - start_time > est_time:
                desired_rot_speed_in_base = np.zeros(3)
            
            current_coordinate = self.get_current_coordinate()
            
            if dist > min_dist and curr_speed > min_speed and error < slowdown_dist:
                curr_speed -= 0.0001
                print("Slowing down, current speed: ", curr_speed, "-", len(data))
            
            trajectory_desired_coordinate = desired_coordinate - current_coordinate
            velocity_desired_coordinate = trajectory_desired_coordinate / np.linalg.norm(trajectory_desired_coordinate) * curr_speed 
            
            velocity_desired = np.concatenate((velocity_desired_coordinate, desired_rot_speed_in_base)) # May need to change from zeros to actual values while creating sphere
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ velocity_desired     
                                
            self.publishVelocityCommand(velocity_command) 
            
            self.ros_rate.sleep()  
            current_coordinate = self.get_current_coordinate()
            tmp_error = np.linalg.norm(desired_coordinate - current_coordinate)
            if tmp_error < slowdown_dist and tmp_error - error > 0.0001:
                print("Error in position, breaking.", np.linalg.norm(desired_coordinate - current_coordinate), "-", error)
                break
            error = np.linalg.norm(desired_coordinate - current_coordinate)
                        
        self.stop_movement()        
        print("Position reached, current position: ", current_coordinate)

    def follow_trajectory_in_base(self, trajectory, speed=0.05):
        
        print("Following trajectory with", len(trajectory), "points.")
        
        if len(trajectory) == 0:
            self.stop_movement()
            print("Empty trajectory.")
            return
        
        for point in trajectory:
            print("Point: ", point)
            self.go_to_pose_in_base(point[0], point[1], speed)
            if len(point) > 2 and point[2] == "close":
                self.gripper.close()
            elif len(point) > 2 and point[2] == "open":
                self.gripper.open()
        
    def go_to_joint_pos(self, desired_joint_positions, speed=0.05):
        
        trajectory = desired_joint_positions - self.joint_states_global["pos"]      
        norm = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])        
        delta_t = norm/speed
                
        velocity_command = trajectory / delta_t
        print("desired_joint_positions: ", desired_joint_positions)
        print("velocity_command: ", velocity_command)
            
        self.publishVelocityCommand(velocity_command)
        
        start_time = time.time()
        
        while norm > 0.0005 and time.time() - start_time < delta_t + 0.1:
            norm = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])
            
        self.stop_movement() 
        
        print("Position reached, current position: ", self.joint_states_global["pos"])  
       
    # there is no reason to use this instead of cubic, but I will keep it just in case
    def pick_up_object(self, object_name, speed=0.05):
        
        print("Picking up object: ", object_name)
        
        try:
            object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
            object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            while object_position[2] > 0.5:
                self.init_tf()
                object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
                object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            object_position[2] += self.gripper_length + 0.02        
            object_position[0] -= 0.01
            object_position[1] -= 0.01
            self.gripper.open_async()
            
            eef_to_desired = self.tfBuffer.lookup_transform(self.eef, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([eef_to_desired.transform.rotation.x,
                                    eef_to_desired.transform.rotation.y,
                                    eef_to_desired.transform.rotation.z,
                                    eef_to_desired.transform.rotation.w]).as_rotvec()
            
            pick_to_base = self.tfBuffer.lookup_transform(self.base, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([pick_to_base.transform.rotation.x,
                                                        pick_to_base.transform.rotation.y,
                                                        pick_to_base.transform.rotation.z,
                                                        pick_to_base.transform.rotation.w]).apply(rotvec)
            
            trajectory = []
            trajectory.append((object_position,rotvec))
            trajectory.append((object_position - np.array([0, 0, 0.05]),None,"close"))
            trajectory.append((object_position,None))
            
            self.follow_trajectory_in_base(trajectory, speed)
        except Exception as e:
            print("Error in picking up object: ", e)
            return
   
    def follow_derivative(self, x_derivative, y_derivative, z_derivative, trajectory_length, desired_orientation, speed):
                            
        est_time = trajectory_length / speed
        if desired_orientation is not None:
            rot_speed = np.linalg.norm(desired_orientation)/(est_time)           
            desired_rot_speed = (desired_orientation / np.linalg.norm(desired_orientation)) * rot_speed 
            print("Desired Rot Speed: ", desired_rot_speed)
        else:
            print("No desired orientation, keeping current orientation.")
            desired_rot_speed = np.zeros(3)
                
        t = 0
        speed_multiplier = 1
        
        while rclpy.ok() and t <= 1:
            
            if t < 0.25:
                speed_multiplier = 0.2 + 0.8*(t*4)
            elif t > 0.75:
                speed_multiplier = 0.2 + 0.8*((1-t)*4)
            else:
                speed_multiplier = 1
                                            
            trajectory_desired = np.array([x_derivative(t), y_derivative(t), z_derivative(t)])
            desired_linear_speed = (trajectory_desired / np.linalg.norm(trajectory_desired)) * speed
            desired_speed = np.concatenate((desired_linear_speed, desired_rot_speed))
            desired_speed *= speed_multiplier
            
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ desired_speed
            self.publishVelocityCommand(velocity_command)
            
            self.ros_rate.sleep()
            t += (speed * speed_multiplier) / (trajectory_length * self.control_rate)            
                            
        self.stop_movement()
    
    def set_pick_up_cubic_variables(self, object_name):
        print("Setting up cubic pick up variables.")
        
        try:
            object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
            object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            while object_position[2] > 0.5:
                self.init_tf()
                object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
                object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            object_position[2] += self.gripper_length + 0.02        
            object_position[0] -= 0.01
            object_position[1] -= 0.01
            self.gripper.open_async()
            
            eef_to_desired = self.tfBuffer.lookup_transform(self.eef, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([eef_to_desired.transform.rotation.x,
                                    eef_to_desired.transform.rotation.y,
                                    eef_to_desired.transform.rotation.z,
                                    eef_to_desired.transform.rotation.w]).as_rotvec()
            
            pick_to_base = self.tfBuffer.lookup_transform(self.base, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([pick_to_base.transform.rotation.x,
                                                        pick_to_base.transform.rotation.y,
                                                        pick_to_base.transform.rotation.z,
                                                        pick_to_base.transform.rotation.w]).apply(rotvec)
                                    
            start_point = self.get_current_coordinate()
            mid_point = object_position
            end_point = object_position - np.array([0, 0, 0.05])
            x_coeff, y_coeff, z_coeff = three_point_cubic_spline(start_point, mid_point, end_point)
       
            x_coordinate = lambda t: x_coeff[0]*t**3 + x_coeff[1]*t**2 + x_coeff[2]*t + x_coeff[3]
            y_coordinate = lambda t: y_coeff[0]*t**3 + y_coeff[1]*t**2 + y_coeff[2]*t + y_coeff[3]
            z_coordinate = lambda t: z_coeff[0]*t**3 + z_coeff[1]*t**2 + z_coeff[2]*t + z_coeff[3]
            
            length_local = 0
            prev_point = np.array([x_coordinate(0), y_coordinate(0), z_coordinate(0)])
            i = 0
            while i < 1000:
                point = np.array([x_coordinate(i/1000), y_coordinate(i/1000), z_coordinate(i/1000)])
                length_local += np.linalg.norm(point - prev_point)
                prev_point = point      
                i += 1          
                        
            est_time = length_local / self.speed
            if rotvec is not None:
                rot_speed = np.linalg.norm(rotvec)/(est_time)           
                desired_rot_speed_local = (rotvec / np.linalg.norm(rotvec)) * rot_speed 
                print("Desired Rot Speed: ", desired_rot_speed_local)
            else:
                print("No desired orientation, keeping current orientation.")
                desired_rot_speed_local = np.zeros(3)
                         
            self.desired_rot_speed = desired_rot_speed_local
                                    
            self.length = length_local
            
            self.x_derivative = lambda t: 3*x_coeff[0]*t**2 + 2*x_coeff[1]*t + x_coeff[2]
            self.y_derivative = lambda t: 3*y_coeff[0]*t**2 + 2*y_coeff[1]*t + y_coeff[2]
            self.z_derivative = lambda t: 3*z_coeff[0]*t**2 + 2*z_coeff[1]*t + z_coeff[2]
                            
            self.t = 0
            
            self.speed_multiplier = 1
                
        except Exception as e:
            print("Error in setting pick up object cubically:", e)
            return
    
    def set_place_cubic_variables(self):
        
        try:
                        
            start_point = self.get_current_coordinate()
            end_point = start_point + np.array([-0.8, -0.5, 0])
            mid_point = (start_point + end_point)/2 + np.array([0, 0, 0.5])
            x_coeff, y_coeff, z_coeff = three_point_cubic_spline(start_point, mid_point, end_point)                         
            
            x_coordinate = lambda t: x_coeff[0]*t**3 + x_coeff[1]*t**2 + x_coeff[2]*t + x_coeff[3]
            y_coordinate = lambda t: y_coeff[0]*t**3 + y_coeff[1]*t**2 + y_coeff[2]*t + y_coeff[3]
            z_coordinate = lambda t: z_coeff[0]*t**3 + z_coeff[1]*t**2 + z_coeff[2]*t + z_coeff[3]
            
            length_local = 0
            prev_point = np.array([x_coordinate(0), y_coordinate(0), z_coordinate(0)])
            i = 0
            while i < 1000:
                point = np.array([x_coordinate(i/1000), y_coordinate(i/1000), z_coordinate(i/1000)])
                length_local += np.linalg.norm(point - prev_point)
                prev_point = point  
                i += 1              
            
            self.desired_rot_speed = np.zeros(3)
                                    
            self.length = length_local
            
            self.x_derivative = lambda t: 3*x_coeff[0]*t**2 + 2*x_coeff[1]*t + x_coeff[2]
            self.y_derivative = lambda t: 3*y_coeff[0]*t**2 + 2*y_coeff[1]*t + y_coeff[2]
            self.z_derivative = lambda t: 3*z_coeff[0]*t**2 + 2*z_coeff[1]*t + z_coeff[2]
                            
            self.t = 0
            
            self.speed_multiplier = 1
                
        except Exception as e:
            print("Error in picking up object cubically:", e)
            return
    
    def set_breathing_gazing(self):
        
        self.go_to_home_pos(self.speed) 
        self.gripper.close_async()                          
        self.breathe_controller.reset()
    
    def get_derivative_velocities(self):
                    
        if self.t < 0.25:
            self.speed_multiplier = 0.2 + 0.8*(self.t*4)
        elif self.t > 0.75:
            self.speed_multiplier = 0.2 + 0.8*((1-self.t)*4)
        else:
            self.speed_multiplier = 1
                                        
        trajectory_desired = np.array([self.x_derivative(self.t), self.y_derivative(self.t), self.z_derivative(self.t)])
        desired_linear_speed = (trajectory_desired / np.linalg.norm(trajectory_desired)) * self.speed
        desired_speed = np.concatenate((desired_linear_speed, self.desired_rot_speed))
        desired_speed *= self.speed_multiplier
        
        pinv_jacobian = self.get_inverse_jacobian()
        velocity_command = pinv_jacobian @ desired_speed
        
        self.t += (self.speed * self.speed_multiplier) / (self.length * self.control_rate)            
        
        return velocity_command
                                 
    def pick_up_object_cubic(self, object_name, speed=0.05):
        print("Picking up object cubically: ", object_name)
        
        try:
            object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
            object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            while object_position[2] > 0.5:
                self.init_tf()
                object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
                object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        
            object_position[2] += self.gripper_length + 0.02        
            object_position[0] -= 0.01
            object_position[1] -= 0.01
            self.gripper.open_async()
            
            eef_to_desired = self.tfBuffer.lookup_transform(self.eef, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([eef_to_desired.transform.rotation.x,
                                    eef_to_desired.transform.rotation.y,
                                    eef_to_desired.transform.rotation.z,
                                    eef_to_desired.transform.rotation.w]).as_rotvec()
            
            pick_to_base = self.tfBuffer.lookup_transform(self.base, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([pick_to_base.transform.rotation.x,
                                                        pick_to_base.transform.rotation.y,
                                                        pick_to_base.transform.rotation.z,
                                                        pick_to_base.transform.rotation.w]).apply(rotvec)
                        
            start_point = self.get_current_coordinate()
            mid_point = object_position
            end_point = object_position - np.array([0, 0, 0.05])
            x_coeff, y_coeff, z_coeff = three_point_cubic_spline(start_point, mid_point, end_point)                         
            
            x_derivative = lambda t: 3*x_coeff[0]*t**2 + 2*x_coeff[1]*t + x_coeff[2]
            y_derivative = lambda t: 3*y_coeff[0]*t**2 + 2*y_coeff[1]*t + y_coeff[2]
            z_derivative = lambda t: 3*z_coeff[0]*t**2 + 2*z_coeff[1]*t + z_coeff[2]
            
            x_coordinate = lambda t: x_coeff[0]*t**3 + x_coeff[1]*t**2 + x_coeff[2]*t + x_coeff[3]
            y_coordinate = lambda t: y_coeff[0]*t**3 + y_coeff[1]*t**2 + y_coeff[2]*t + y_coeff[3]
            z_coordinate = lambda t: z_coeff[0]*t**3 + z_coeff[1]*t**2 + z_coeff[2]*t + z_coeff[3]
            
            length = 0
            prev_point = np.array([x_coordinate(0), y_coordinate(0), z_coordinate(0)])
            for i in range(1000):
                point = np.array([x_coordinate(i/1000), y_coordinate(i/1000), z_coordinate(i/1000)])
                length += np.linalg.norm(point - prev_point)
                prev_point = point                
            
            """traj_points = MarkerArray()
            id = 0
            print("Generating trajectory points.")
            points = np.linspace(0, 1, num=100) # for visualization
            for pt in points:
                
                marker = Marker()
                marker.header.frame_id = self.base
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "trajectory"
                marker.id = id
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.pose.position.x = x_coordinate(pt)
                marker.pose.position.y = y_coordinate(pt)
                marker.pose.position.z = z_coordinate(pt)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                                
                traj_points.markers.append(marker)
                
                id += 1
            print("Trajectory points: ", len(traj_points.markers))
            self.traj_publisher.publish(traj_points)    
            print("Trajectory published.")"""   
            
            self.follow_derivative(x_derivative, y_derivative, z_derivative, length, rotvec, speed)
            
            self.gripper.close()
                
        except Exception as e:
            print("Error in picking up object cubically:", e)
            return
    
    def place_object_cubic(self, speed=0.05):
        
        try:
                        
            start_point = self.get_current_coordinate()
            end_point = start_point + np.array([-0.8, -0.5, 0])
            mid_point = (start_point + end_point)/2 + np.array([0, 0, 0.5])
            x_coeff, y_coeff, z_coeff = three_point_cubic_spline(start_point, mid_point, end_point)                         
            
            x_derivative = lambda t: 3*x_coeff[0]*t**2 + 2*x_coeff[1]*t + x_coeff[2]
            y_derivative = lambda t: 3*y_coeff[0]*t**2 + 2*y_coeff[1]*t + y_coeff[2]
            z_derivative = lambda t: 3*z_coeff[0]*t**2 + 2*z_coeff[1]*t + z_coeff[2]
            
            x_coordinate = lambda t: x_coeff[0]*t**3 + x_coeff[1]*t**2 + x_coeff[2]*t + x_coeff[3]
            y_coordinate = lambda t: y_coeff[0]*t**3 + y_coeff[1]*t**2 + y_coeff[2]*t + y_coeff[3]
            z_coordinate = lambda t: z_coeff[0]*t**3 + z_coeff[1]*t**2 + z_coeff[2]*t + z_coeff[3]
            
            length = 0
            prev_point = np.array([x_coordinate(0), y_coordinate(0), z_coordinate(0)])
            for i in range(1000):
                point = np.array([x_coordinate(i/1000), y_coordinate(i/1000), z_coordinate(i/1000)])
                length += np.linalg.norm(point - prev_point)
                prev_point = point                
            
            """traj_points = MarkerArray()
            id = 0
            print("Generating trajectory points.")
            points = np.linspace(0, 1, num=100) # for visualization
            for pt in points:
                
                marker = Marker()
                marker.header.frame_id = self.base
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "trajectory"
                marker.id = id
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.pose.position.x = x_coordinate(pt)
                marker.pose.position.y = y_coordinate(pt)
                marker.pose.position.z = z_coordinate(pt)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.01
                marker.scale.y = 0.01
                marker.scale.z = 0.01
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                                
                traj_points.markers.append(marker)
                
                id += 1
            print("Trajectory points: ", len(traj_points.markers))
            self.traj_publisher.publish(traj_points)    
            print("Trajectory published.")"""   
            
            self.follow_derivative(x_derivative, y_derivative, z_derivative, length, None, speed)
            
            self.gripper.open()
                
        except Exception as e:
            print("Error in picking up object cubically:", e)
            return
    
    def go_to_home_pos(self, speed=0.05):
        self.go_to_joint_pos(self.home_pos, speed=speed)
        
    def stop_movement(self):
        vel_msg = Float64MultiArray()
        vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
        self.velocityControllerPub.publish(vel_msg)
        self.vel_comm_ind = 0
        print("Movement stopped.")
          
    def get_jacobian_matrix(self):
        # get_jacobian is a function that returns the jacobian matrix of the robot at the current joint states, it was calculated using matlab.
        jacobian = get_jacobian(self.joint_states_global["pos"].tolist())
        return jacobian
                        
    def get_inverse_jacobian(self):
        # get_jacobian is a function that returns the jacobian matrix of the robot at the current joint states, it was calculated using matlab.
        jacobian = get_jacobian(self.joint_states_global["pos"].tolist())
        invj = np.linalg.pinv(jacobian, rcond=1e-15)
        return invj
        
    def filter_joint_velocities(self, joint_velocities):
        
        if self.vel_comm_ind == 0:
            self.prev_velocities = joint_velocities
            self.vel_comm_ind = 1
            return joint_velocities
                
        max_speed_diffs = np.array([0.004, 0.004, 0.004, 0.004, 0.004, 0.004])
        critical_speed_diffs = np.array([0.007, 0.007, 0.007, 0.007, 0.007, 0.007])
        
        filter_weight = 0.8
                
        for j in range(self.num_of_total_joints):
            change = joint_velocities[j] - self.prev_velocities[j]
            if abs(change) > max_speed_diffs[j]:
                joint_velocities[j] = self.prev_velocities[j]*filter_weight + joint_velocities[j]*(1-filter_weight)
                #joint_velocities[j] = self.prev_velocities[j] + np.sign(change) * max_speed_diffs[j]
                if abs(joint_velocities[j] - self.prev_velocities[j]) > critical_speed_diffs[j]:
                    joint_velocities[j] = self.prev_velocities[j] + np.sign(change) * critical_speed_diffs[j] 
        
        self.prev_velocities = joint_velocities
        
        return joint_velocities
    
    def publishVelocityCommand(self, vels):
        if type(vels) is not np.ndarray:
            vels = np.array(vels)
         
        vels = self.filter_joint_velocities(vels)      
        
        data.append(self.joint_states_global["vels"])
        data_c.append(vels)
            
        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
    def joint_state_callback(self, msg):
        # Why does UR5e always send joint states in an order other than 012345? It was 210345 in ROS1 and now this.
        self.joint_states_global["pos"] = np.array([
                                            msg.position[5],
                                            msg.position[0],
                                            msg.position[1],
                                            msg.position[2],
                                            msg.position[3],
                                            msg.position[4]])
        self.joint_states_global["vels"] = np.array([
                                            msg.velocity[5],
                                            msg.velocity[0],
                                            msg.velocity[1],
                                            msg.velocity[2],
                                            msg.velocity[3],
                                            msg.velocity[4]])
        self.joint_states_global["effs"] = np.array([
                                            msg.effort[5],
                                            msg.effort[0],
                                            msg.effort[1],
                                            msg.effort[2],
                                            msg.effort[3],
                                            msg.effort[4]])         

    def raise_eef(self, raise_amount, speed=0.1):
                
        t = 0
        speed_multiplier = 1
        
        trajectory_desired = np.array([0,0,1*np.sign(raise_amount)])
        raise_amount = abs(raise_amount)
        
        while rclpy.ok() and t <= 1:
            
            if t < 0.25:
                speed_multiplier = 0.2 + 0.8*(t*4)
            elif t > 0.75:
                speed_multiplier = 0.2 + 0.8*((1-t)*4)
            else:
                speed_multiplier = 1
                                            
            desired_linear_speed = (trajectory_desired / np.linalg.norm(trajectory_desired)) * speed
            desired_speed = np.concatenate((desired_linear_speed, np.zeros(3)))
            desired_speed *= speed_multiplier
            
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ desired_speed
            self.publishVelocityCommand(velocity_command)
            
            self.ros_rate.sleep()
            t += (speed * speed_multiplier) / (raise_amount * self.control_rate)            
                            
        self.stop_movement()
        
    def state_machine(self):
        
        # Set flags for breathing and gazing
        do_breathing = True
        do_gazing = True
        
        inf_breathe = False
        
        start_time = time.time()
        
        while rclpy.ok():
            
            if self.state == RobotState.START:
                
                self.go_to_home_pos(speed=self.speed)
                self.set_breathing_gazing()
                self.state = RobotState.BREATHING_GAZING
                start_time = time.time()
                continue
                
            elif self.state == RobotState.BREATHING_GAZING:
                
                breathing_velocities = np.zeros(self.num_of_breathing_joints)
                if do_breathing:
                    breathing_velocities = self.breathe_controller.step(self.joint_states_global["pos"], self.joint_states_global["vels"], get_jacobian)
                
                gazing_velocities = np.zeros(self.num_of_total_joints - self.num_of_breathing_joints)
                if do_gazing:                
                    gazing_velocities = self.get_gaze_velocities(breathing_velocities)                
                        
                velocity_command = np.concatenate((breathing_velocities, gazing_velocities))
                
                if not inf_breathe and time.time() - start_time > 10:
                    self.state = RobotState.PICKING
                    self.set_pick_up_cubic_variables("marker_11")
                    continue
                
            elif self.state == RobotState.PICKING:
                
                if self.t >= 0 and self.t <= 1:
                    velocity_command = self.get_derivative_velocities()
                elif self.t < 0:
                    print("SOMETHING WENT EXTREMELY WRONG.")
                    self.stop_movement()
                    self.state = RobotState.END
                    continue
                else:
                    self.stop_movement()
                    self.state = RobotState.PLACING
                    self.set_place_cubic_variables()
                    self.gripper.close()
                    continue
                
            elif self.state == RobotState.PLACING:
                
                if self.t >= 0 and self.t <= 1:
                    velocity_command = self.get_derivative_velocities()
                elif self.t < 0:
                    print("SOMETHING WENT EXTREMELY WRONG.")
                    self.stop_movement()
                    self.state = RobotState.END
                    continue
                else:
                    self.stop_movement()
                    self.gripper.open()
                    self.set_breathing_gazing()
                    self.state = RobotState.BREATHING_GAZING
                    inf_breathe = True
                    continue
                
            elif self.state == RobotState.END:
                
                self.stop_movement()
                break
                
            self.publishVelocityCommand(velocity_command)
                
            self.ros_rate.sleep()
            
        self.stop_movement()
        self.go_to_home_pos(speed=self.speed)

    def deneme(self):
        
        while rclpy.ok():
            
            eef_tf = self.tfBuffer.lookup_transform(self.eef, "pick_pose", rclpy.time.Time())
            rotvec = R.from_quat([eef_tf.transform.rotation.x,
                                    eef_tf.transform.rotation.y,
                                    eef_tf.transform.rotation.z,
                                    eef_tf.transform.rotation.w]).as_rotvec()
            
            desired_rot_speed = (rotvec / np.linalg.norm(rotvec)) * 0.1
            pick_to_base = self.tfBuffer.lookup_transform(self.base, "pick_pose", rclpy.time.Time())
            desired_rot_speed_in_base = R.from_quat([pick_to_base.transform.rotation.x,
                                                        pick_to_base.transform.rotation.y,
                                                        pick_to_base.transform.rotation.z,
                                                        pick_to_base.transform.rotation.w]).apply(desired_rot_speed)
            
            print("Desired Rot Speed: ", desired_rot_speed_in_base)
                
            desired_vel = np.concatenate((np.zeros(3), desired_rot_speed_in_base))
            
            desired_command = self.get_inverse_jacobian() @ desired_vel
            self.publishVelocityCommand(desired_command)
            
            self.ros_rate.sleep()
        
def spin_thread(node):
    
    try:
        rclpy.spin(node)
    except Exception as e:
        node.stop_movement()
        pass
        
def main(args=None):

    try:
        rclpy.init(args=args)    
        new_controller = NewController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(new_controller,), daemon=True)
        controller_spin_thread.start()
        
        print("Starting main.")
        
        new_controller.speed = 0.3
        new_controller.state_machine()
        
    except Exception as e:    
        print("Error in main: ", e)   
        pass
    
    try:
        new_controller.stop_movement()
                
        new_controller.destroy_node()
        rclpy.shutdown()
    except:
        pass
    
    controller_spin_thread.join()
    
    print("\n take care of yourself \n")
    
    j1 = [x for x in np.array(data)[:,0]]
    j2 = [x for x in np.array(data)[:,1]]
    j3 = [x for x in np.array(data)[:,2]]
    j4 = [x for x in np.array(data)[:,3]]
    j5 = [x for x in np.array(data)[:,4]]
    j6 = [x for x in np.array(data)[:,5]]
    
    plt.plot(j1, label="Joint 1")
    plt.plot(j2, label="Joint 2")
    plt.plot(j3, label="Joint 3")
    plt.plot(j4, label="Joint 4")
    plt.plot(j5, label="Joint 5")
    plt.plot(j6, label="Joint 6")
    plt.legend()
    plt.show()
    
    j1 = [x for x in np.array(data_c)[:,0]]
    j2 = [x for x in np.array(data_c)[:,1]]
    j3 = [x for x in np.array(data_c)[:,2]]
    j4 = [x for x in np.array(data_c)[:,3]]
    j5 = [x for x in np.array(data_c)[:,4]]
    j6 = [x for x in np.array(data_c)[:,5]]
    
    plt.plot(j1, label="Joint 1")
    plt.plot(j2, label="Joint 2")
    plt.plot(j3, label="Joint 3")
    plt.plot(j4, label="Joint 4")
    plt.plot(j5, label="Joint 5")
    plt.plot(j6, label="Joint 6")
    plt.legend()
    plt.show()
    