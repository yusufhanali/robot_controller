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

from .jacobian.jacobian_src import get_jacobian    

from .breathing import breathing_src as breathing


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
        self.control_rate =500  # Hz
        self.ros_rate = self.create_rate(self.control_rate, self.get_clock())
        
        self.base = "base"
        self.eef = "wrist_3_link"
        self.world = "world"
        
        self.init_joint_states()
        
        self.init_velocity_controller()
        
        self.init_tf()
        
        self.init_gripper()
        
        self.init_breather()
        
        self.vel_comm_ind = 0
        
        self.prev_velocities = np.zeros(6)
        self.prev_accelerations = np.zeros(6)
        
        self.home_pos = np.array([-0.8, -1.73, -2.1,  0.73,  1.52,  3.16])
        self.gripper_length = 0.17
                                        
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('present time.')  
    
    def get_gaze_velocities(self, breathing_velocities):
        
        tf_name = "eye"
                    
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
        
        self.gazing_breathing_compensation = False
        self.use_helmet = False
                    
        self.breathe_controller.reset()    
            
        while rclpy.ok():              
            
            breathing_velocities = np.zeros(self.num_of_breathing_joints)
            if do_breathing:
                breathing_velocities, mag = self.breathe_controller.step(self.joint_states_global["pos"], self.joint_states_global["vels"], get_jacobian)
            
            gazing_velocities = np.zeros(self.num_of_total_joints - self.num_of_breathing_joints)
            if do_gazing:                
                gazing_velocities = self.get_gaze_velocities(breathing_velocities)                
                    
            # Publish joint vels to robot
            velocity_command = np.concatenate((breathing_velocities, gazing_velocities))
                                 
            self.publishVelocityCommand(velocity_command)
                                                
            self.ros_rate.sleep()
                        
        print("Exiting breathe and gaze.")
            
        self.stop_movement() 
     
    def follow_trajectory_in_base(self, trajectory, speed=0.05):
        
        if len(trajectory) == 0:
            self.stop_movement()
            print("Empty trajectory.")
            return
        
        current_pos = np.zeros(3)        
        transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
        current_pos[0] = transformation.transform.translation.x
        current_pos[1] = transformation.transform.translation.y
        current_pos[2] = transformation.transform.translation.z 
        
        max_error = 0.0005
        
        amount_points = len(trajectory)
        
        pos_index = 1
        
        self.go_to_point(trajectory[0], speed)
        
        print("Following trajectory with", len(trajectory), "points.")
                    
        while rclpy.ok() and pos_index < amount_points:

            transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
            current_pos[0] = transformation.transform.translation.x
            current_pos[1] = transformation.transform.translation.y
            current_pos[2] = transformation.transform.translation.z 
            
            if np.linalg.norm(current_pos - trajectory[pos_index]) < max_error:
                pos_index -= 1
            
            trajectory_desired_pos = trajectory[pos_index+1] - current_pos
            trajectory_desired_full = np.concatenate((trajectory_desired_pos, np.zeros(3))) # May need to change from zeros to actual values while creating sphere
            velocity_desired = trajectory_desired_full / np.linalg.norm(trajectory_desired_full) * speed
            pinv_jacobian = self.get_inverse_jacobian()                
            velocity_command = pinv_jacobian @ velocity_desired                    
            self.publishVelocityCommand(velocity_command)
             
            pos_index += 1            
            self.ros_rate.sleep()  
                                    
        self.stop_movement()        
        print("Trajectory followed, current position: ", current_pos)

    def go_to_pose_in_base(self, desired_coordinate, desired_orientation, speed=0.05):       
        # currently pose is a name in the tf_tree
        
        current_coordinate = np.zeros(3)        
        transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
        current_coordinate[0] = transformation.transform.translation.x
        current_coordinate[1] = transformation.transform.translation.y
        current_coordinate[2] = transformation.transform.translation.z 
        
        max_error = 0.001
        
        if np.linalg.norm(desired_coordinate - current_coordinate) < max_error:
            self.stop_movement()
            print("Already at the desired position.")
            return
                                 
        print("Current position: ", current_coordinate)   
        print("Going to point: ", desired_coordinate)
        
        est_time = np.linalg.norm(desired_coordinate - current_coordinate) / speed
        
        eef_to_desired = self.tfBuffer.lookup_transform(self.eef, desired_orientation, rclpy.time.Time())
        rotvec = R.from_quat([eef_to_desired.transform.rotation.x,
                                eef_to_desired.transform.rotation.y,
                                eef_to_desired.transform.rotation.z,
                                eef_to_desired.transform.rotation.w]).as_rotvec()
        rot_speed = np.linalg.norm(rotvec)/est_time
        
        desired_rot_speed = (rotvec / np.linalg.norm(rotvec)) * rot_speed
        pick_to_base = self.tfBuffer.lookup_transform(self.base, desired_orientation, rclpy.time.Time())
        desired_rot_speed_in_base = R.from_quat([pick_to_base.transform.rotation.x,
                                                    pick_to_base.transform.rotation.y,
                                                    pick_to_base.transform.rotation.z,
                                                    pick_to_base.transform.rotation.w]).apply(desired_rot_speed)
        
        print("Desired Rot Speed: ", desired_rot_speed_in_base)
                                    
        while rclpy.ok() and np.linalg.norm(desired_coordinate - current_coordinate) > max_error:

            transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
            current_coordinate[0] = transformation.transform.translation.x
            current_coordinate[1] = transformation.transform.translation.y
            current_coordinate[2] = transformation.transform.translation.z 
            
            trajectory_desired_coordinate = desired_coordinate - current_coordinate
            velocity_desired_coordinate = trajectory_desired_coordinate / np.linalg.norm(trajectory_desired_coordinate) * speed 
            
            velocity_desired = np.concatenate((velocity_desired_coordinate, desired_rot_speed_in_base)) # May need to change from zeros to actual values while creating sphere
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ velocity_desired     
                                
            self.publishVelocityCommand(velocity_command) 
            
            self.ros_rate.sleep()  
                        
        self.stop_movement()        
        print("Position reached, current position: ", current_coordinate)

    def go_to_point_in_base(self, desired_coordinate, speed=0.05):
        
        current_coordinate = np.zeros(3)        
        transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
        current_coordinate[0] = transformation.transform.translation.x
        current_coordinate[1] = transformation.transform.translation.y
        current_coordinate[2] = transformation.transform.translation.z 
        
        max_error = 0.001
        
        if np.linalg.norm(desired_coordinate - current_coordinate) < max_error:
            self.stop_movement()
            print("Already at the desired position.")
            return
                                 
        print("Current position: ", current_coordinate)   
        print("Going to point: ", desired_coordinate)
                                    
        while rclpy.ok() and np.linalg.norm(desired_coordinate - current_coordinate) > max_error:

            transformation = self.tfBuffer.lookup_transform(self.base, self.eef, rclpy.time.Time())
            current_coordinate[0] = transformation.transform.translation.x
            current_coordinate[1] = transformation.transform.translation.y
            current_coordinate[2] = transformation.transform.translation.z 
            
            trajectory_desired_coordinate = desired_coordinate - current_coordinate
            trajectory_desired_full = np.concatenate((trajectory_desired_coordinate, np.zeros(3))) # May need to change from zeros to actual values while creating sphere
            velocity_desired = trajectory_desired_full / np.linalg.norm(trajectory_desired_full) * speed
            pinv_jacobian = self.get_inverse_jacobian()
            velocity_command = pinv_jacobian @ velocity_desired                  
            self.publishVelocityCommand(velocity_command) 
            
            self.ros_rate.sleep()  
                        
        self.stop_movement()        
        print("Position reached, current position: ", current_coordinate)
     
    def go_to_point_in_world(self, desired_coordinate, speed=0.05):
        
        world_to_base = self.tfBuffer.lookup_transform(self.base, self.world, rclpy.time.Time())
        world_to_base_point = R.from_quat([world_to_base.transform.rotation.x,
                                            world_to_base.transform.rotation.y,
                                            world_to_base.transform.rotation.z,
                                            world_to_base.transform.rotation.w]).apply(desired_coordinate)
        world_to_base_point[0] += world_to_base.transform.translation.x
        world_to_base_point[1] += world_to_base.transform.translation.y
        world_to_base_point[2] += world_to_base.transform.translation.z
        
        self.go_to_point_in_base(world_to_base_point, speed)
        
    def go_to_joint_pos(self, desired_joint_positions, speed=0.05):
        
        trajectory = desired_joint_positions - self.joint_states_global["pos"]      
        norm = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])        
        delta_t = norm/speed
                
        velocity_command = trajectory / delta_t
        print("desired_joint_positions: ", desired_joint_positions)
        print("velocity_command: ", velocity_command)
            
        self.publishVelocityCommand(velocity_command)
        
        start_time = time.time()
        
        while norm > 0.001 and time.time() - start_time < delta_t + 0.1:
            norm = np.linalg.norm(desired_joint_positions - self.joint_states_global["pos"])
            
        self.stop_movement() 
        
        print("Position reached, current position: ", self.joint_states_global["pos"])  
       
    def pick_up_object(self, object_name, speed=0.05):
        
        print("Picking up object: ", object_name)
        
        try:
            object_tf = self.tfBuffer.lookup_transform(self.base, object_name, rclpy.time.Time())
            object_position = np.array([object_tf.transform.translation.x, object_tf.transform.translation.y, object_tf.transform.translation.z])
        except TransformException as e:
            print("Error in getting object position: ", e)
            return
        
        object_position[2] += self.gripper_length + 0.02
        
        object_position[0] -= 0.01
        object_position[1] -= 0.01
        self.gripper.open_async()
        
        self.go_to_pose_in_base(object_position, "pick_pose", speed)
        
        self.go_to_point_in_base(object_position - np.array([0, 0, 0.05]), speed)
        
        self.gripper.close()
        
        self.go_to_point_in_base(object_position, speed)
    
    def go_to_home_pos(self, speed=0.05):
        self.go_to_joint_pos(self.home_pos, speed=speed)
        
    def stop_movement(self):
        vel_msg = Float64MultiArray()
        vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
        self.velocityControllerPub.publish(vel_msg)
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
        elif self.vel_comm_ind == 1:
            self.prev_accelerations = (joint_velocities - self.prev_velocities) * self.control_rate
            self.prev_velocities = joint_velocities
            self.vel_comm_ind = 2
            return joint_velocities
        
        curr_accelerations = (joint_velocities - self.prev_velocities) * self.control_rate
        
        max_acceleration_diffs = np.array([99, 99, 99, 2, 2, 2])
        
        filter_weight = 0.7
        
        for i in range(self.num_of_total_joints):
            if abs(curr_accelerations[i] - self.prev_accelerations[i]) > max_acceleration_diffs[i]:
                joint_velocities[i] = self.prev_velocities[i]*filter_weight + joint_velocities[i]*(1-filter_weight)
                #joint_velocities[i] = self.prev_velocities[i] + (np.sign(curr_accelerations[i] - self.prev_accelerations[i]) * max_acceleration_diffs[i] / self.control_rate)
                self.prev_accelerations[i] = (joint_velocities[i] - self.prev_velocities[i]) * self.control_rate
        
        self.prev_velocities = joint_velocities
        self.prev_accelerations = curr_accelerations
        
        return joint_velocities
    
    def publishVelocityCommand(self, vels):
        if type(vels) is not np.ndarray:
            vels = np.array(vels)
         
        vels = self.filter_joint_velocities(vels)            
            
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
    except:
        node.stop_movement()
        pass
        
def main(args=None):

    try:
        rclpy.init(args=args)    
        new_controller = NewController()
        
        controller_spin_thread = threading.Thread(target=spin_thread, args=(new_controller,), daemon=True)
        controller_spin_thread.start()
        
        print("Starting main.")
        
        #new_controller.go_to_home_pos(speed=0.1)
        #new_controller.breathe_and_gaze()
        
        new_controller.pick_up_object("marker_11")
    except:    
        print("Error in main.")   
        pass
    
    try:
        new_controller.stop_movement()
                
        new_controller.destroy_node()
        rclpy.shutdown()
    except:
        pass
    
    controller_spin_thread.join()
    
    print("\n take care of yourself \n")
    