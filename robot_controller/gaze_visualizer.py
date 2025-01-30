import rclpy
import numpy as np

from skspatial.objects import Line, Plane
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from .util import order_points_counter_clockwise

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

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
            cross = np.cross(table_points[i+1] - table_points[i], tar - table_points[i+1])
            
            if first_cross is None:
                first_cross = cross
            else:
                # if the cross product is ever in the opposite direction of the first cross product, return False since that implies the point is outside the table.
                #Note: the cross product can be either in the exacy same direction or in the exact opposite direction, never in between.
                if np.dot(first_cross, cross) < 0:
                    return False, tar

        return True, tar

    except ValueError:
        # No valid intersection found
        return False, None

def get_table_points(table_pos, table_rot, x_width, z_breadth):
    
    table_points = []
    
    table_points.append(table_rot.apply([x_width, 0, z_breadth]) + table_pos)
    table_points.append(table_rot.apply([-x_width, 0, z_breadth]) + table_pos)
    table_points.append(table_rot.apply([-x_width, 0, -z_breadth]) + table_pos)
    table_points.append(table_rot.apply([x_width, 0, -z_breadth]) + table_pos)
    
    return np.array(table_points)

class GazeViz(Node):

    def __init__(self):
        super().__init__('gaze_visualizer')
        self.get_logger().info('gaze visualizer Node Started')
        
        self.gaze_vector_topic = "gaze_vector"
        self.subscriber = self.create_subscription(Float64MultiArray, self.gaze_vector_topic, self.gaze_callback, 10)
        
        self.table_points = None
        
        self.gaze_target_pub = self.create_publisher(Float64MultiArray, 'gaze_target', 10)
        self.gaze_pub = self.create_publisher(MarkerArray, 'gaze_marker_topic', 10)
        self.table_pub = self.create_publisher(MarkerArray, 'table_marker_topic', 10)
        self.table_timer = self.create_timer(0.01, self.table_callback)
        
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)
        
        while not self.tfBuffer.can_transform("base", "wrist_3_link", rclpy.time.Time()):
            #self.get_logger().info('Waiting for tf tree...')
            rclpy.spin_once(self)
        self.get_logger().info('TF tree received.')
    
        
    def table_callback(self):
        
        try:
            markers = MarkerArray()
                
            #TABLE 160*70 height-72 z-80 x-35 
                            
            table_transform = self.tfBuffer.lookup_transform("world", "rigid_body_3", rclpy.time.Time())
            r = R.from_quat([table_transform.transform.rotation.x, table_transform.transform.rotation.y, table_transform.transform.rotation.z, table_transform.transform.rotation.w])
            table_pos = np.array([table_transform.transform.translation.x, table_transform.transform.translation.y, table_transform.transform.translation.z])
            self.table_points = get_table_points(table_pos, r, 0.35, 0.8)
                            
            table_pt = Marker()
            table_pt.header.frame_id = "world"
            table_pt.header.stamp = self.get_clock().now().to_msg()
            table_pt.id = 0
            table_pt.type = Marker.CUBE
            table_pt.action = Marker.ADD
            table_pt.pose.position.x = table_pos[0]
            table_pt.pose.position.y = table_pos[1]
            table_pt.pose.position.z = table_pos[2]
            table_pt.scale.x = 0.7
            table_pt.scale.y = 0.001
            table_pt.scale.z = 1.6
            table_pt.color.a = 1.0
            table_pt.color.r = 1.0
            table_pt.color.g = 1.0
            table_pt.color.b = 0.0
            table_pt.pose.orientation.w = table_transform.transform.rotation.w
            table_pt.pose.orientation.x = table_transform.transform.rotation.x
            table_pt.pose.orientation.y = table_transform.transform.rotation.y
            table_pt.pose.orientation.z = table_transform.transform.rotation.z
            
            markers.markers.append(table_pt)            
            
            self.table_pub.publish(markers)
        except:
            self.get_logger().info("Error in table callback")
        
    def gaze_callback(self, msg):
        
        use_helmet = True
        gaze_markers = MarkerArray()
        
        try:                        
            head_frame = "eye" if use_helmet else "exp/head"
                        
            eye_transform = self.tfBuffer.lookup_transform("world", head_frame, rclpy.time.Time())
            gaze_tail = np.array([eye_transform.transform.translation.x, eye_transform.transform.translation.y, eye_transform.transform.translation.z])
                        
            msg_data = msg.data
            gaze_direction = np.array(msg_data)
            gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
                        
            camera_transform = self.tfBuffer.lookup_transform("world", "camera_link", rclpy.time.Time())
            quat = [camera_transform.transform.rotation.x, camera_transform.transform.rotation.y, camera_transform.transform.rotation.z, camera_transform.transform.rotation.w]
            Rot = R.from_quat(quat)
            gaze_direction = Rot.apply(gaze_direction)
                                    
            gaze_success, gaze_target = compute_gaze_target(gaze_direction, gaze_tail, self.table_points)
                        
            if gaze_target is not None:            
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
                
                gaze_markers.markers.append(gaze_marker)
                
                gaze_arrow = Marker()
                gaze_arrow.header.frame_id = "world"
                gaze_arrow.header.stamp = self.get_clock().now().to_msg()
                gaze_arrow.id = 1
                gaze_arrow.type = Marker.ARROW
                gaze_arrow.action = Marker.ADD
                tail_point = Point()
                tail_point.x = gaze_tail[0]
                tail_point.y = gaze_tail[1]
                tail_point.z = gaze_tail[2]
                end_ = gaze_tail + gaze_direction
                end_point = Point()
                end_point.x = end_[0]
                end_point.y = end_[1]
                end_point.z = end_[2]
                gaze_arrow.points.append(tail_point)
                gaze_arrow.points.append(end_point)
                gaze_arrow.scale.x = 0.01
                gaze_arrow.scale.y = 0.02
                gaze_arrow.scale.z = 0.0
                gaze_arrow.color.a = 1.0
                gaze_arrow.color.r = 1.0
                gaze_arrow.color.g = 0.0
                gaze_arrow.color.b = 0.0
                gaze_arrow.pose.orientation.w = 1.0
                gaze_arrow.pose.orientation.x = 0.0
                gaze_arrow.pose.orientation.y = 0.0
                gaze_arrow.pose.orientation.z = 0.0
                
                gaze_markers.markers.append(gaze_arrow)
                                
            self.gaze_pub.publish(gaze_markers)
            self.gaze_target_pub.publish(Float64MultiArray(data=gaze_target))
                        
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