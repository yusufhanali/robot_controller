import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState


joint_states_global = {}
    

class NewController(Node):

    def __init__(self):
        super().__init__('new_controller')
        self.get_logger().info('New Controller Node Started')
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        
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
    rclpy.init(args=args)
    new_controller = NewController()
    rclpy.spin(new_controller)
    new_controller.destroy_node()
    rclpy.shutdown()