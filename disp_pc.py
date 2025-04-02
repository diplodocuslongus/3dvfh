import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

def disparity_to_3d(disparity, f, B, cx, cy):
    """
    Converts a disparity image to 3D points using NumPy.

    Args:
        disparity (numpy.ndarray): Disparity image (2D array).
        f (float): Focal length of the camera.
        B (float): Baseline (distance between the stereo cameras).
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.

    Returns:
        numpy.ndarray: Nx3 array of 3D points.
    """
    # Get the image dimensions
    h, w = disparity.shape

    # Create a grid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Avoid division by zero by masking invalid disparity values
    valid_mask = disparity > 0

    # Compute depth (Z)
    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid_mask] = (f * B) / (disparity[valid_mask]*0.125)

    # Compute X and Y
    X = (x_coords - cx) * Z / f
    Y = (y_coords - cy) * Z / f

    # Stack X, Y, Z into an Nx3 array of 3D points
    points_3d = np.stack((Z[valid_mask], -X[valid_mask], -Y[valid_mask]), axis=-1)

    return points_3d

def disp_callback(img_msg):
    #print(type(img_msg.data))
    p_3d = disparity_to_3d(np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width), 470.051*0.25, 0.0750492, 314.96*0.25, 229.359*0.25)
    header = Header()
    header.frame_id = "body"
    header.stamp = node.get_clock().now().to_msg()
    pc_pub.publish(point_cloud2.create_cloud_xyz32(header, p_3d))

rclpy.init()
node = rclpy.create_node('disp_pc')

best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
pc_pub = node.create_publisher(PointCloud2, "obstacles", 1)
disp_sub = node.create_subscription(Image, "disparity", disp_callback, qos_profile=best_effort_qos)

while rclpy.ok():
    try:
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        break
rclpy.try_shutdown()
print("bye")
