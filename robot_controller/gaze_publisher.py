import rclpy

import socket
import os

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

SOCKET_FILE = '/tmp/usta_sockets_gaze.sock'

class GazePublisher(Node):

    def __init__(self):
        super().__init__('gaze_publisher')
        self.get_logger().info('gaze publisher Node Started')
        
        self.gaze_vector_topic = "gaze_vector"
        self.publisher = self.create_publisher(Float64MultiArray, self.gaze_vector_topic, 10)
        
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        self.server.bind(SOCKET_FILE)

        self.server.listen(1)
        print("Started listenin for gaze vectors...")
        
        self.timer = self.create_timer(1/100, self.get_gaze)       
        
        
    def get_gaze(self):
        
        self.timer.cancel()
        
        while True:
            connection, _ = self.server.accept()
            data = connection.recv(1024)
            
            decoded_data = data.decode('utf-8') 
            xyz = decoded_data.split(" ")

            gaze_msg = Float64MultiArray()
            gaze_msg.data = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            self.publisher.publish(gaze_msg)
            
            connection.close()
        
def main(args=None):
    
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)

    try:
        rclpy.init(args=args)    
        gazepub = GazePublisher()
        rclpy.spin(gazepub)    
    except KeyboardInterrupt:
        pass
    gazepub.destroy_node()
    rclpy.shutdown()