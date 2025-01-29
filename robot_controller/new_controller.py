import rclpy
import rclpy.exceptions
import rclpy.executors
from rclpy.node import Node

import threading

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


data = []


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
        
        period, amplitude = 2, 1.5 #get_session_parameters(session_speed)
        freq = 1.0 / period  # Named as beta in the paper
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
        
        self.vel_comm_ind = 0
        
        self.prev_velocities = np.zeros(6)
        self.prev_accelerations = np.zeros(6)
                                        
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
        delta = instant_velocities[:3] * 2
        
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
            head_in_w1_pos = np.array([head_in_w1.transform.translation.x-delta_in_w1[0], head_in_w1.transform.translation.y-delta_in_w1[1], 0])
            head_in_w1_norm = np.linalg.norm(head_in_w1_pos)
            cw1 = np.arccos(0.0997 / head_in_w1_norm) - np.arccos(np.dot([0,-1,0], head_in_w1_pos/head_in_w1_norm))
            cw1 = cw1*(-1) if head_in_w1_pos[0] > 0 else cw1
            gazing_velocities[0] = cw1*gaze_multiplier
            
            head_in_w2 = self.tfBuffer.lookup_transform("wrist_2_link", tf_name, rclpy.time.Time(seconds=0))
            head_in_w2_pos = np.array([head_in_w2.transform.translation.x-delta_in_w2[0], head_in_w2.transform.translation.y-delta_in_w2[1], 0])
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
        
    def stop_movement(self):
        self.publishVelocityCommand([0.0]*6)
          
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
        data.append(vels)
            
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
        
        new_controller.breathe_and_gaze()
        
    except KeyboardInterrupt:
            
        controller_spin_thread.join()
                
        print("\n take care of yourself \n")
        pass
    
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
    
    new_controller.destroy_node()
    #rclpy.shutdown()