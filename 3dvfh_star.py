import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

import numpy as np
import math

def vfh_star_3d_pointcloud_target_direction(point_cloud, target_direction, bin_size=10, max_range=4.0, safety_distance=1.0, alpha=0.5, valley_threshold=0.1):
    """
    Implements a simplified 3D Vector Field Histogram* (VFH*) for UAV obstacle avoidance,
    using 3D point cloud and a target direction vector as input, returning a normalized 3D vector.

    Args:
        point_cloud (numpy.ndarray): Nx3 array of 3D points (x, y, z).
        target_direction (numpy.ndarray): 3D vector representing the target direction.
        bin_size (int): Size of each histogram bin in degrees.
        max_range (float): Maximum range of the sensor.
        safety_distance (float): Minimum safe distance from obstacles.
        alpha (float): Weighting factor for target direction influence (0 to 1).
        valley_threshold (float): Threshold for identifying valleys in the histogram.

    Returns:
        numpy.ndarray: Normalized 3D vector representing the best direction.
    """

    # Normalize the target direction.
    normalized_direction = target_direction / np.linalg.norm(target_direction)

    # Convert normalized direction to yaw and pitch.
    pitch_target = math.asin(normalized_direction[2])
    yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

    # 1. Histogram Creation
    histogram = np.zeros((360 // bin_size, 180 // bin_size)) # yaw x pitch

    for point in point_cloud:
        x = point.x
        y = point.y
        z = point.z

        depth = math.sqrt(x**2 + y**2 + z**2)

        if 0 < depth < max_range:
            # Convert to spherical coordinates
            yaw = math.atan2(y, x)
            pitch = math.atan2(z, math.sqrt(x**2 + y**2))

            # Calculate magnitude (influence) of the obstacle.
            magnitude = (safety_distance / depth) ** 2

            # Bin the obstacle into the histogram
            yaw_bin = int((math.degrees(yaw) + 180) // bin_size) % (360 // bin_size)
            pitch_bin = int((math.degrees(pitch) + 90) // bin_size) % (180 // bin_size)

            histogram[yaw_bin, pitch_bin] += magnitude

    # 2. Polar Histogram Reduction (Valley Detection)
    # (Simplified valley detection)

    valley_mask = np.zeros_like(histogram, dtype=bool)

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            # Check for local minima (valleys).
            if (
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin - 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin - 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < valley_threshold
            ):
                valley_mask[yaw_bin, pitch_bin] = True

    # 3. Target Direction Selection (VFH* Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            # VFH* cost function: obstacle density + weighted distance from target, prioritize valleys.
            cost = histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)
            if valley_mask[yaw_bin, pitch_bin]:
                cost = cost * 0.5 # lower cost for valley bins, encourage valley following.

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    # Convert spherical coordinates to a 3D vector.
    x = math.cos(best_pitch) * math.cos(best_yaw)
    y = math.cos(best_pitch) * math.sin(best_yaw)
    z = math.sin(best_pitch)

    # Normalize the vector.
    vector = np.array([x, y, z])
    normalized_vector = vector / np.linalg.norm(vector)

    return normalized_vector

def vfh_star_3d_pointcloud_positions(point_cloud, current_position, destination_position, bin_size=10, max_range=4.0, safety_distance=1.0, alpha=0.5, valley_threshold=0.1):
    """
    Implements a simplified 3D Vector Field Histogram* (VFH*) for UAV obstacle avoidance,
    using 3D point cloud, current position, and destination position as input, returning a normalized 3D vector.

    Args:
        point_cloud (numpy.ndarray): Nx3 array of 3D points (x, y, z).
        current_position (numpy.ndarray): 3D vector representing the UAV's current position.
        destination_position (numpy.ndarray): 3D vector representing the UAV's destination position.
        bin_size (int): Size of each histogram bin in degrees.
        max_range (float): Maximum range of the sensor.
        safety_distance (float): Minimum safe distance from obstacles.
        alpha (float): Weighting factor for target direction influence (0 to 1).
        valley_threshold (float): Threshold for identifying valleys in the histogram.

    Returns:
        numpy.ndarray: Normalized 3D vector representing the best direction.
    """

    # Calculate the target direction vector.
    target_direction = destination_position - current_position
    normalized_direction = target_direction / np.linalg.norm(target_direction)

    # Convert normalized direction to yaw and pitch.
    pitch_target = math.asin(normalized_direction[2])
    yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

    # 1. Histogram Creation
    histogram = np.zeros((360 // bin_size, 180 // bin_size)) # yaw x pitch

    for point in point_cloud:
        x = point.x
        y = point.y
        z = point.z

        depth = math.sqrt(x**2 + y**2 + z**2)

        if 0 < depth < max_range:
            # Convert to spherical coordinates
            yaw = math.atan2(y, x)
            pitch = math.atan2(z, math.sqrt(x**2 + y**2))

            # Calculate magnitude (influence) of the obstacle.
            magnitude = (safety_distance / depth) ** 2

            # Bin the obstacle into the histogram
            yaw_bin = int((math.degrees(yaw) + 180) // bin_size) % (360 // bin_size)
            pitch_bin = int((math.degrees(pitch) + 90) // bin_size) % (180 // bin_size)

            histogram[yaw_bin, pitch_bin] += magnitude

    # 2. Polar Histogram Reduction (Valley Detection)
    # (Simplified valley detection)

    valley_mask = np.zeros_like(histogram, dtype=bool)

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            # Check for local minima (valleys).
            if (
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin - 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin - 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < valley_threshold
            ):
                valley_mask[yaw_bin, pitch_bin] = True

    # 3. Target Direction Selection (VFH* Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            # VFH* cost function: obstacle density + weighted distance from target, prioritize valleys.
            cost = histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)
            if valley_mask[yaw_bin, pitch_bin]:
                cost = cost * 0.5 # lower cost for valley bins, encourage valley following.

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    # Convert spherical coordinates to a 3D vector.
    x = math.cos(best_pitch) * math.cos(best_yaw)
    y = math.cos(best_pitch) * math.sin(best_yaw)
    z = math.sin(best_pitch)

    # Normalize the vector.
    vector = np.array([x, y, z])
    normalized_vector = vector / np.linalg.norm(vector)

    return normalized_vector

def vfh_plus_3d_pointcloud(point_cloud, normalized_direction, bin_size=10, max_range=5.0, safety_distance=1.0, alpha=0.5):
    """
    Implements a simplified 3D Vector Field Histogram+ (VFH+) for UAV obstacle avoidance,
    using 3D point cloud and a normalized 3D direction vector as input, returning a normalized 3D vector.

    Args:
        point_cloud (numpy.ndarray): Nx3 array of 3D points (x, y, z).
        normalized_direction (numpy.ndarray): Normalized 3D vector representing the target direction.
        bin_size (int): Size of each histogram bin in degrees.
        max_range (float): Maximum range of the sensor.
        safety_distance (float): Minimum safe distance from obstacles.
        alpha (float): Weighting factor for target direction influence (0 to 1).

    Returns:
        numpy.ndarray: Normalized 3D vector representing the best direction.
    """

    # Input vector is already normalized.

    # Convert normalized direction to yaw and pitch.
    pitch_target = math.asin(normalized_direction[2])
    yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

    # 1. Histogram Creation
    histogram = np.zeros((360 // bin_size, 180 // bin_size)) # yaw x pitch

    for point in point_cloud:
        x = point.x
        y = point.y
        z = point.z

        depth = math.sqrt(x**2 + y**2 + z**2)

        if 0 < depth < max_range:
            # Convert to spherical coordinates
            yaw = math.atan2(y, x)
            pitch = math.atan2(z, math.sqrt(x**2 + y**2))

            # Calculate magnitude (influence) of the obstacle.
            magnitude = (safety_distance / depth) ** 2

            # Bin the obstacle into the histogram
            yaw_bin = int((math.degrees(yaw) + 180) // bin_size) % (360 // bin_size)
            pitch_bin = int((math.degrees(pitch) + 90) // bin_size) % (180 // bin_size)

            histogram[yaw_bin, pitch_bin] += magnitude

    # 2. Polar Histogram Reduction (Simplified)

    # 3. Target Direction Selection (VFH+ Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            # VFH+ cost function: obstacle density + weighted distance from target.
            cost = histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    # Convert spherical coordinates to a 3D vector.
    x = math.cos(best_pitch) * math.cos(best_yaw)
    y = math.cos(best_pitch) * math.sin(best_yaw)
    z = math.sin(best_pitch)

    # Normalize the vector.
    vector = np.array([x, y, z])
    normalized_vector = vector / np.linalg.norm(vector)

    return normalized_vector

def vfh_3d_pointcloud(point_cloud, normalized_direction, bin_size=10, max_range=5.0, safety_distance=1.0):
    """
    Implements a simplified 3D Vector Field Histogram (VFH*) for UAV obstacle avoidance,
    using 3D point cloud and a normalized 3D direction vector as input, returning a normalized 3D vector.

    Args:
        point_cloud (numpy.ndarray): Nx3 array of 3D points (x, y, z).
        normalized_direction (numpy.ndarray): Normalized 3D vector representing the target direction.
        bin_size (int): Size of each histogram bin in degrees.
        max_range (float): Maximum range of the sensor.
        safety_distance (float): Minimum safe distance from obstacles.

    Returns:
        numpy.ndarray: Normalized 3D vector representing the best direction.
    """

    # Input vector is already normalized, so we can use it directly.

    # Convert normalized direction to yaw and pitch.
    pitch_target = math.asin(normalized_direction[2])
    yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

    # 1. Histogram Creation
    histogram = np.zeros((360 // bin_size, 180 // bin_size)) # yaw x pitch

    for point in point_cloud:
        x = point.x
        y = point.y
        z = point.z

        depth = math.sqrt(x**2 + y**2 + z**2)

        if 0 < depth < max_range:
            # Convert to spherical coordinates
            yaw = math.atan2(y, x)
            pitch = math.atan2(z, math.sqrt(x**2 + y**2))

            # Calculate magnitude (influence) of the obstacle.
            magnitude = (safety_distance / depth) ** 2

            # Bin the obstacle into the histogram
            yaw_bin = int((math.degrees(yaw) + 180) // bin_size) % (360 // bin_size)
            pitch_bin = int((math.degrees(pitch) + 90) // bin_size) % (180 // bin_size)

            histogram[yaw_bin, pitch_bin] += magnitude

    # 2. Polar Histogram Reduction (Simplified)

    # 3. Target Direction Selection
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    for yaw_bin in range(360 // bin_size):
        for pitch_bin in range(180 // bin_size):
            cost = histogram[yaw_bin, pitch_bin] + 0.1 * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    # Convert spherical coordinates to a 3D vector.
    x = math.cos(best_pitch) * math.cos(best_yaw)
    y = math.cos(best_pitch) * math.sin(best_yaw)
    z = math.sin(best_pitch)

    # Normalize the vector.
    vector = np.array([x, y, z])
    normalized_vector = vector / np.linalg.norm(vector)

    return normalized_vector

def obs_callback(msg):
    #print(type(msg.points))
    #for p in msg.points:
    #    print(p.x, p.y, p.z)
    global latest_obs
    latest_obs = msg.points

rclpy.init()
node = rclpy.create_node('obs_avd')

# Define a point in the "map" frame
point_in_map = PointStamped()
point_in_map.header.frame_id = "map"
point_in_map.header.stamp = node.get_clock().now().to_msg()
point_in_map.point.x = 5.0
point_in_map.point.y = 0.0
point_in_map.point.z = 1.0

latest_obs = []

# Create a TF2 buffer and listener
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

obs_sub = node.create_subscription(PointCloud, "obstacles", obs_callback, 1)
avd_pub = node.create_publisher(TwistStamped, "avoid_direction", 1)

while rclpy.ok():
    try:
        rclpy.spin_once(node)
        if latest_obs:
            try:
                # Lookup the transform from "map" to "body"
                transform = tf_buffer.lookup_transform(
                    "body",  # Target frame
                    "map",   # Source frame
                    rclpy.time.Time(),  # Use the latest available transform
                    timeout=rclpy.duration.Duration(seconds=0.0)
                )
            except Exception as e:
                print(e)
            else:
                # Transform the point
                point_in_body = do_transform_point(point_in_map, transform)
                avd_dir = vfh_star_3d_pointcloud_target_direction(latest_obs, np.array([point_in_body.point.x, point_in_body.point.y, point_in_body.point.z]))
                m = TwistStamped()
                m.header.frame_id = "body"
                m.header.stamp = node.get_clock().now().to_msg()
                m.twist.linear.x = avd_dir[0]
                m.twist.linear.y = avd_dir[1]
                m.twist.linear.z = avd_dir[2]
                avd_pub.publish(m)
                latest_obs = []
    except KeyboardInterrupt:
        break
rclpy.try_shutdown()
print("bye")


