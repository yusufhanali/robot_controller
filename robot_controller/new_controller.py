import rclpy
from rclpy.node import Node

import numpy as np

from .robotiq_gripper import RobotiqGripper as robotiq_gripper

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .jacobian import get_jacobian    

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
            self.get_logger().info('Waiting for joint states...')
            rclpy.spin_once(self)
        self.get_logger().info('Joint states received.')
        
    def init_velocity_controller(self):
        # Publish velocity commands
        self.velocityControllerTopic = "forward_velocity_controller/commands" # All movements should be done w.r.t. "base", not "base_link" or "world"
        self.velocityControllerPub = self.create_publisher(Float64MultiArray, self.velocityControllerTopic, 10)
        
        while not self.velocityControllerPub.get_subscription_count():
            self.get_logger().info('Waiting for velocity controller to connect...')
            rclpy.spin_once(self)
        self.get_logger().info('Velocity controller connected.')

    def init_tf(self):
        # Get tf tree
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        while not self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()):
            self.get_logger().info('Waiting for tf tree...')
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

    def __init__(self):
        super().__init__('new_controller')
        # Sanity check at the start of node (The phrases are very unprofessionally taken from my favorite anime, Serial Experiments Lain)
        self.get_logger().info('Present day,')
        
        self.init_joint_states()
        
        self.init_velocity_controller()
        
        self.init_tf()
        
        self.init_gripper()
        
        # Initialize control rate
        self.control_rate = 500
        self.ros_rate = self.create_rate(self.control_rate)
        
        # This is my easy emektar debugging and implementation method. (If you dont know what emektar is, please contact me at e2522100@ceng.metu.edu.tr I would be happy to explain / or look up a dictionary. The choice is, as it always is, yours.)
        self.timer = self.create_timer(1/1000, self.emektar)
                        
        # Sanity check at the end of node. If both of these are printed, then the node is probably working properly.
        self.get_logger().info('present time.')  
        
    def emektar(self):
        
        self.timer.cancel()
        
        self.gripper.close()
                        
    def get_inverse_jacobian(self):
        # get_jacobian is a function that returns the jacobian matrix of the robot at the current joint states, it was calculated using matlab.
        jacobian = get_jacobian(self.joint_states_global["pos"][0], self.joint_states_global["pos"][1], self.joint_states_global["pos"][2], self.joint_states_global["pos"][3], self.joint_states_global["pos"][4], self.joint_states_global["pos"][5])
        invj = np.linalg.pinv(jacobian, rcond=1e-15)
        return invj
        
    def publishVelocityCommand(self, vels):
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
        
        
def main(args=None):

    try:
        rclpy.init(args=args)    
        new_controller = NewController()
        rclpy.spin(new_controller)    
    except KeyboardInterrupt:
        print("\n take care of yourself \n")
        pass
    
    new_controller.destroy_node()
    #rclpy.shutdown()