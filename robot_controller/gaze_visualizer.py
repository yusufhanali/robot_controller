import rclpy
import numpy as np

from skspatial.objects import Line, Plane
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from .util import order_points_counter_clockwise

from visualization_msgs.msg import Marker   

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

TABLE_POINTS = np.array([ [0.69, 0.97, 0.72], [0.03, -0.63, 0.72], [0.00, 0.96, 0.72], [0.72, -0.61, 0.72] ])

def compute_gaze_target(gd, gt, table_points):
    """
    Compute the gaze target location (tar) as the intersection of the table plane and the gaze line.

    Parameters:
        gd (array-like): Gaze direction vector (3D, shape (3,))
        gt (array-like): Gaze line tail (3D, shape (3,))
        table_points (array-like): 4 points of the table plane (shape (4,3))

    Returns:
        tuple:
            found (bool): True if a valid gaze target is found, False otherwise.
            tar (np.ndarray or None): The gaze target location (3D) if found, otherwise None.
    """
    # Test cases (no longer valid since q has been replaced by table_points):

    # Case 1, no intersection (empty set):
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [0, 0, 1]  # Gaze tail
    # n = [1, -1, 0]  # Table normal
    # q = [10, 0, 0]  # A point on the table plane

    # Case 2, the intersection is the whole line:
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [0, 0, 0]  # Gaze tail
    # n = [1, -1, 0]  # Table normal
    # q = [0, 0, 0]  # A point on the table plane

    # Case 3, intersection is a point, but an invalid one:
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [2, 2, 2]  # Gaze tail
    # n = [1, 1, 1]  # Table normal
    # q = [0, 0, 0]  # A point on the table plane

    # Case 4, intersection is a valid point:
    # Should return True, [0,0,0]
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [-2, -2, -2]  # Gaze tail
    # n = [1, 1, 1]  # Table normal
    # q = [0.0, 0.0, 0.0]  # A point on the table plane

    try:
        table_points = order_points_counter_clockwise(table_points)
        table_center = np.mean(table_points, axis=0)
        table_n = np.cross(table_points[0] - table_center, table_points[1] - table_center)
        
        # Create Line and Plane objects using the provided parameters
        line = Line(point=gt, direction=gd)
        plane = Plane(point=table_points[0], normal=table_n)

        # Attempt to find the intersection point
        intersection_point = plane.intersect_line(line)

        # Convert the Point object to a NumPy array
        tar = np.array(intersection_point)
        
        # if gaze target is on the opposite direction of the gaze vector, return False.
        if np.dot(tar - gt, gd) <= 0.0:
            return False, None
        
        # Check if the intersection point is inside the table polygon                
        first_cross = None
        
        for i in range(len(table_points) - 1):
            cross = np.cross(table_points[i+1] - table_points[i], table_center - table_points[i+1])
            
            if first_cross is None:
                first_cross = cross
            else:
                # if the cross product is ever in the opposite direction of the first cross product, return False since that implies the point is outside the table.
                #Note: the cross product can be either in the exacy same direction or in the exact opposite direction, never in between.
                if np.dot(first_cross, cross) < 0:
                    return False, None

        return True, tar

    except ValueError:
        # No valid intersection found
        return False, None

class GazeViz(Node):

    def __init__(self):
        super().__init__('gaze_visualizer')
        self.get_logger().info('gaze visualizer Node Started')
        
        self.gaze_vector_topic = "gaze_vector"
        self.subscriber = self.create_subscription(Float64MultiArray, self.gaze_vector_topic, self.gaze_callback, 10)
        
        self.gaze_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
    def gaze_callback(self, msg):
        
        try:
                        
            eye_transform = self.tfBuffer.lookup_transform("world", "eye", rclpy.time.Time())
            gaze_tail = np.array([eye_transform.transform.translation.x, eye_transform.transform.translation.y, eye_transform.transform.translation.z])
                        
            msg_data = msg.data
            gaze_direction = np.array(msg_data)
                        
            camera_transform = self.tfBuffer.lookup_transform("world", "camera_link", rclpy.time.Time())
            quat = [camera_transform.transform.rotation.x, camera_transform.transform.rotation.y, camera_transform.transform.rotation.z, camera_transform.transform.rotation.w]
            Rot = R.from_quat(quat)
            gaze_direction = Rot.apply(gaze_direction)
                                    
            gaze_success, gaze_target = compute_gaze_target(gaze_direction, gaze_tail, TABLE_POINTS)
                        
            if gaze_success:            
                gaze_marker = Marker()
                gaze_marker.header.frame_id = "world"
                gaze_marker.header.stamp = self.get_clock().now().to_msg()
                gaze_marker.id = 0
                gaze_marker.type = Marker.SPHERE
                gaze_marker.action = Marker.ADD
                gaze_marker.pose.position.x = gaze_target[0]
                gaze_marker.pose.position.y = gaze_target[1]
                gaze_marker.pose.position.z = gaze_target[2]
                gaze_marker.scale.x = 0.1
                gaze_marker.scale.y = 0.1
                gaze_marker.scale.z = 0.1
                gaze_marker.color.a = 1.0
                gaze_marker.color.r = 1.0
                gaze_marker.color.g = 0.0
                gaze_marker.color.b = 0.0
                gaze_marker.pose.orientation.w = 1.0
                gaze_marker.pose.orientation.x = 0.0
                gaze_marker.pose.orientation.y = 0.0
                gaze_marker.pose.orientation.z = 0.0
                
                self.gaze_pub.publish(gaze_marker)
                        
        except:
            self.get_logger().info("Error in gaze callback")
        
def main(args=None):

    try:
        rclpy.init(args=args)    
        gazeviz = GazeViz()
        rclpy.spin(gazeviz)    
    except KeyboardInterrupt:
        pass
    gazeviz.destroy_node()
    rclpy.shutdown()