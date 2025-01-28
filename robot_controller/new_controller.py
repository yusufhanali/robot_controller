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

    def __init__(self):
        super().__init__('new_controller')
        self.get_logger().info('New Controller Node Started')
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        self.joint_states_global = {}
        
        self.velocityControllerTopic = "forward_velocity_controller/commands" # All movements should be done w.r.t. "base", not "base_link" or "world"
        self.velocityControllerPub = self.create_publisher(Float64MultiArray, self.velocityControllerTopic, 10)
        
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        self.gripper = GripperController("192.168.1.102", 63352)
        
        self.timer = self.create_timer(1/1000, self.maain)
                        
        print("sanity")        
        
    def maain(self):
        
        self.timer.cancel()
        
        self.gripper.close()
                        
    def get_inverse_jacobian(self):
        jacobian = get_jacobian(self.joint_states_global["pos"][0], self.joint_states_global["pos"][1], self.joint_states_global["pos"][2], self.joint_states_global["pos"][3], self.joint_states_global["pos"][4], self.joint_states_global["pos"][5])
        invj = np.linalg.pinv(jacobian, rcond=1e-15)
        return invj
        
    def publishVelocityCommand(self, vels):
        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
    def joint_state_callback(self, msg):
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
        pass
    new_controller.destroy_node()
    rclpy.shutdown()