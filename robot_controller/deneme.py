import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class GazePublisher(Node):

    def __init__(self):
        super().__init__('gaze_publisher')
        self.get_logger().info('gaze publisher Node Started')
        
        self.gaze_vector_topic = "gaze_vector"
        self.subscriber = self.create_subscription(Float64MultiArray, self.gaze_vector_topic, self.gaze_callback, 10)
        
    def gaze_callback(self, msg):
        print(msg.data)        
        
def main(args=None):

    try:
        rclpy.init(args=args)    
        gazepub = GazePublisher()
        rclpy.spin(gazepub)    
    except KeyboardInterrupt:
        pass
    gazepub.destroy_node()
    rclpy.shutdown()