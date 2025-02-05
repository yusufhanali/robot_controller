import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray


CWD = "/home/kovan/USTA/gaze_test"


class GazeTestDataCollecter(Node):

    def __init__(self):
        super().__init__('gaze_test_data_collecter')
        self.get_logger().info('gaze test data collecter Node Started')
        
        self.gaze_target_sub = self.create_subscription(Float64MultiArray, "gaze_target", self.gaze_target_callback, 10)
        self.gaze_direction_sub = self.create_subscription(Float64MultiArray, "gaze_vector", self.gaze_direction_callback, 10)
        
        self.gaze_directions = []
        self.gaze_targets = []
        self.boxes = []
        self.eyes = []
        
        self.mocap_timer = self.create_timer(0.01, self.mocap_callback)

        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        while not self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()):
            #self.get_logger().info('Waiting for tf tree...')
            rclpy.spin_once(self)
        self.get_logger().info('TF tree received.')
        
        
    def appendWithTimestamp(self, v, data, timestamp=None):
        """
        v: vector, shape of (1,3)
        """
        # Get the current timestamp
        timestamp = self.get_clock().now().to_msg().sec * 1000 + self.get_clock().now().to_msg().nanosec // 1_000_000
        # Append the timestamp and gaze vector as a new row
        new_row = [timestamp, *v]
        data.append(new_row)
        
    def gaze_target_callback(self, msg):
        return self.appendWithTimestamp(np.array(msg.data), self.gaze_targets)
    
    def gaze_direction_callback(self, msg):
        try:
            camera_transform = self.tfBuffer.lookup_transform("world", "camera_link", rclpy.time.Time())
            quat = [camera_transform.transform.rotation.x, camera_transform.transform.rotation.y, camera_transform.transform.rotation.z, camera_transform.transform.rotation.w]
            Rot = R.from_quat(quat)
            gaze_direction = np.array(msg.data)
            gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
            gaze_direction = Rot.apply(gaze_direction)
            print("gaze_direction_callback before return: ", gaze_direction)
            return self.appendWithTimestamp(gaze_direction, self.gaze_directions)
        except:
            self.get_logger().info("error in gaze_direction_callback")

    
    def mocap_callback(self):
        try:
            box = self.tfBuffer.lookup_transform("world", "rigid_body_4", rclpy.time.Time()).transform.translation
            box = np.array([box.x, box.y, box.z])
            self.appendWithTimestamp(box, self.boxes)
            
            eye = self.tfBuffer.lookup_transform("world", "eye", rclpy.time.Time()).transform.translation
            eye = np.array([eye.x, eye.y, eye.z])
            self.appendWithTimestamp(eye, self.eyes)
        except:
            self.get_logger().info("error in mocap_callback")
        

    def save_to_file(self):
        timestamp = int(time.time() * 1000)
        # Save each data as a .npy file
        np.save(CWD+"/targets/"+str(timestamp)+".npy", np.array(self.gaze_targets))
        np.save(CWD+"/directions/"+str(timestamp)+".npy", np.array(self.gaze_directions))
        np.save(CWD+"/eyes/"+str(timestamp)+".npy", np.array(self.eyes))
        np.save(CWD+"/boxes/"+str(timestamp)+".npy", np.array(self.boxes))
        print (f"Data saved with the timestamp {timestamp}")


def main(args=None):
    rclpy.init(args=args)
    node = GazeTestDataCollecter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure gaze data is saved before shutdown
        node.save_to_file()
        if rclpy.ok():  # Check if the context is still valid
            rclpy.shutdown()
            
