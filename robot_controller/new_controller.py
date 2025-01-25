from time import sleep
import rclpy
import numpy as np

import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .jacobian import get_jacobian

joint_states_global = {}
    

class NewController(Node):

    def __init__(self):
        super().__init__('new_controller')
        self.get_logger().info('New Controller Node Started')
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
        self.velocityControllerTopic = "forward_velocity_controller/commands"
        self.velocityControllerPub = self.create_publisher(Float64MultiArray, self.velocityControllerTopic, 10)
        
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        self.create_timer(1/100, self.get_gaze)
                        
        print("sanity")        
        
    def maain(self):
        vel = np.array([0, 0, 0, 0, 0.01, 0])
        comm = self.get_inverse_jacobian() @ vel
        print(comm)
        self.publishVelocityCommand(comm)
                        
    def get_inverse_jacobian(self):
        global joint_states_global
        jacobian = get_jacobian(joint_states_global["pos"][0], joint_states_global["pos"][1], joint_states_global["pos"][2], joint_states_global["pos"][3], joint_states_global["pos"][4], joint_states_global["pos"][5])
        invj = np.linalg.pinv(jacobian, rcond=1e-15)
        return invj
        
    def publishVelocityCommand(self, vels):
        vel_msg = Float64MultiArray()
        vel_msg.data = vels
        self.velocityControllerPub.publish(vel_msg)
        
    def joint_state_callback(self, msg):
        global joint_states_global
        joint_states_global["pos"] = np.array([
                                            msg.position[5],
                                            msg.position[0],
                                            msg.position[1],
                                            msg.position[2],
                                            msg.position[3],
                                            msg.position[4]])
        joint_states_global["vels"] = np.array([
                                            msg.velocity[5],
                                            msg.velocity[0],
                                            msg.velocity[1],
                                            msg.velocity[2],
                                            msg.velocity[3],
                                            msg.velocity[4]])
        joint_states_global["effs"] = np.array([
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