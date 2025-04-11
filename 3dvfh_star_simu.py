# add visu and for experimenting
# and comments for my understanding

import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

# added, in order to be able to simulate the transformation normally done by VINS
# here VINS is unavailable
from tf2_ros import TransformBroadcaster


logger = rclpy.logging.get_logger("3dvfh_logger")

# TODO use global or rename, to review
bin_sz = 10

def vfh_star_3d_pointcloud_target_direction_memory(point_cloud,prev_histo, target_direction, prv_yaw, prv_pitch, bin_size=10, max_range=4.0, safety_distance=1.0, alpha=0.5, valley_threshold=0.1, prv_weight=0.1):
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
        #x = point.x
        #y = point.y
        #z = point.z
        x, y, z = point

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

            # Add to neighboring
#            hy = (yaw_bin + 1) % (360 // bin_size)
#            hp = (pitch_bin + 1) % (180 // bin_size)
#            sw = magnitude * 0.5
#            histogram[yaw_bin-1, pitch_bin-1] += sw
#            histogram[yaw_bin-1, pitch_bin] += sw
#            histogram[yaw_bin-1, hp] += sw
#            histogram[yaw_bin, pitch_bin-1] += sw
#            histogram[yaw_bin, hp] += sw
#            histogram[hy, pitch_bin-1] += sw
#            histogram[hy, pitch_bin] += sw
#            histogram[hy, hp] += sw

    # Define the yaw range (in degrees)
    yaw_min = -50  # Minimum yaw angle
    yaw_max = 50   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % (360 // bin_size)
    yaw_max_bin = int((yaw_max + 180) // bin_size) % (360 // bin_size)

    # Define the pitch range (in degrees)
    pitch_min = -40  # Minimum pitch angle
    pitch_max = 40   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % (180 // bin_size)
    pitch_max_bin = int((pitch_max + 90) // bin_size) % (180 // bin_size)

    to_inflate = []
    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):  # Restrict pitch_bin to the specified range
            if histogram[yaw_bin, pitch_bin] < valley_threshold:
                big = np.max(np.array((histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin], histogram[yaw_bin - 1, pitch_bin], histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)], histogram[yaw_bin, pitch_bin - 1])))
                if big > valley_threshold and histogram[yaw_bin, pitch_bin] / big < 0.1:
                    to_inflate.append((yaw_bin, pitch_bin, big))
    for ff in to_inflate:
        #print("from", histogram[ff[0], ff[1]], "to", ff[2]*0.8)
        histogram[ff[0], ff[1]] = ff[2]
    #print(len(to_inflate))


    # 2. Polar Histogram Reduction (Valley Detection)
    # (Simplified valley detection)

    valley_mask = np.zeros_like(histogram, dtype=bool)

    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):
            # Check for local minima (valleys).
            if (
                histogram[yaw_bin, pitch_bin] < valley_threshold and
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin - 1, pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, pitch_bin - 1]
            ):
                valley_mask[yaw_bin, pitch_bin] = True

    # Define the yaw range (in degrees)
    yaw_min = -30  # Minimum yaw angle
    yaw_max = 30   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % (360 // bin_size)
    yaw_max_bin = int((yaw_max + 180) // bin_size) % (360 // bin_size)

    # Define the pitch range (in degrees)
    pitch_min = -20  # Minimum pitch angle
    pitch_max = 20   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % (180 // bin_size)
    pitch_max_bin = int((pitch_max + 90) // bin_size) % (180 // bin_size)

    # 3. Target Direction Selection (VFH* Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')
    # TODO : weight of using the previous histo
    wph = 0.5 # cost of obstacle in the previous histogram
    # cost function
    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):  # Restrict pitch_bin to the specified range
            # VFH* cost function: obstacle density + weighted distance from target, prioritize valleys.
            # TODO check here (using prev histo as part of cost function)
            cost = wph * prev_histo[yaw_bin, pitch_bin]+ histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)
            # cost = histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)

            # Favor previous yaw by adding a penalty for deviation from prv_yaw
            if prv_yaw is not None:
                yaw_bin_radians = math.radians(yaw_bin * bin_size - 180)
                cost += prv_weight * abs(yaw_bin_radians - prv_yaw)  # Adjust the weight (0.1) as needed

            if prv_pitch is not None:
                pitch_bin_radians = math.radians(pitch_bin * bin_size - 90)
                cost += prv_weight * abs(pitch_bin_radians - prv_pitch)  # Adjust pitch weight (e.g., 0.1)

            if valley_mask[yaw_bin, pitch_bin]:
                cost = cost * 0.5 # lower cost for valley bins, encourage valley following.

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    # experimenting: add previous histogram
    histogram_prev = histogram.copy()

    return best_yaw, best_pitch,histogram_prev

def median_bin(image, n):
    """
    Perform 5x5 median binning on a mono image.

    Args:
        image (numpy.ndarray): Input 2D mono image.

    Returns:
        numpy.ndarray: Downsampled image after 5x5 median binning.
    """
    # Ensure the image dimensions are divisible by 5
    h, w = image.shape
    h_new, w_new = h // n, w // n
    image = image[:h_new * n, :w_new * n]  # Crop to make divisible by 5

    # Reshape into 5x5 blocks
    reshaped = image.reshape(h_new, n, w_new, n)

    # Compute the median for each 5x5 block
    binned = np.median(reshaped, axis=(1, 3))

    return binned

def vfh_star_3d_pointcloud_target_direction(point_cloud, target_direction, prv_yaw, prv_pitch, bin_size=10, max_range=4.0, safety_distance=1.0, alpha=0.5, valley_threshold=0.1, prv_weight=0.1):
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
        #x = point.x
        #y = point.y
        #z = point.z
        x, y, z = point

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

            # Add to neighboring
#            hy = (yaw_bin + 1) % (360 // bin_size)
#            hp = (pitch_bin + 1) % (180 // bin_size)
#            sw = magnitude * 0.5
#            histogram[yaw_bin-1, pitch_bin-1] += sw
#            histogram[yaw_bin-1, pitch_bin] += sw
#            histogram[yaw_bin-1, hp] += sw
#            histogram[yaw_bin, pitch_bin-1] += sw
#            histogram[yaw_bin, hp] += sw
#            histogram[hy, pitch_bin-1] += sw
#            histogram[hy, pitch_bin] += sw
#            histogram[hy, hp] += sw

    # Define the yaw range (in degrees)
    yaw_min = -50  # Minimum yaw angle
    yaw_max = 50   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % (360 // bin_size)
    yaw_max_bin = int((yaw_max + 180) // bin_size) % (360 // bin_size)

    # Define the pitch range (in degrees)
    pitch_min = -40  # Minimum pitch angle
    pitch_max = 40   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % (180 // bin_size)
    pitch_max_bin = int((pitch_max + 90) // bin_size) % (180 // bin_size)

    to_inflate = []
    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):  # Restrict pitch_bin to the specified range
            if histogram[yaw_bin, pitch_bin] < valley_threshold:
                big = np.max(np.array((histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin], histogram[yaw_bin - 1, pitch_bin], histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)], histogram[yaw_bin, pitch_bin - 1])))
                if big > valley_threshold and histogram[yaw_bin, pitch_bin] / big < 0.1:
                    to_inflate.append((yaw_bin, pitch_bin, big))
    for ff in to_inflate:
        #print("from", histogram[ff[0], ff[1]], "to", ff[2]*0.8)
        histogram[ff[0], ff[1]] = ff[2]
    #print(len(to_inflate))


    # 2. Polar Histogram Reduction (Valley Detection)
    # (Simplified valley detection)

    valley_mask = np.zeros_like(histogram, dtype=bool)

    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):
            # Check for local minima (valleys).
            if (
                histogram[yaw_bin, pitch_bin] < valley_threshold and
                histogram[yaw_bin, pitch_bin] < histogram[(yaw_bin + 1) % (360 // bin_size), pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin - 1, pitch_bin] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, (pitch_bin + 1) % (180 // bin_size)] and
                histogram[yaw_bin, pitch_bin] < histogram[yaw_bin, pitch_bin - 1]
            ):
                valley_mask[yaw_bin, pitch_bin] = True

    # Define the yaw range (in degrees)
    yaw_min = -30  # Minimum yaw angle
    yaw_max = 30   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % (360 // bin_size)
    yaw_max_bin = int((yaw_max + 180) // bin_size) % (360 // bin_size)

    # Define the pitch range (in degrees)
    pitch_min = -20  # Minimum pitch angle
    pitch_max = 20   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % (180 // bin_size)
    pitch_max_bin = int((pitch_max + 90) // bin_size) % (180 // bin_size)

    # 3. Target Direction Selection (VFH* Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % (360 // bin_size)
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % (180 // bin_size)

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    # cost function
    for yaw_bin in range(yaw_min_bin, yaw_max_bin + 1):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin + 1):  # Restrict pitch_bin to the specified range
            # VFH* cost function: obstacle density + weighted distance from target, prioritize valleys.
            cost = histogram[yaw_bin, pitch_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)

            # Favor previous yaw by adding a penalty for deviation from prv_yaw
            if prv_yaw is not None:
                yaw_bin_radians = math.radians(yaw_bin * bin_size - 180)
                cost += prv_weight * abs(yaw_bin_radians - prv_yaw)  # Adjust the weight (0.1) as needed

            if prv_pitch is not None:
                pitch_bin_radians = math.radians(pitch_bin * bin_size - 90)
                cost += prv_weight * abs(pitch_bin_radians - prv_pitch)  # Adjust pitch weight (e.g., 0.1)

            if valley_mask[yaw_bin, pitch_bin]:
                cost = cost * 0.5 # lower cost for valley bins, encourage valley following.

            if cost < min_cost:
                min_cost = cost
                best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90)

    return best_yaw, best_pitch

def median_bin(image, n):
    """
    Perform 5x5 median binning on a mono image.

    Args:
        image (numpy.ndarray): Input 2D mono image.

    Returns:
        numpy.ndarray: Downsampled image after 5x5 median binning.
    """
    # Ensure the image dimensions are divisible by 5
    h, w = image.shape
    h_new, w_new = h // n, w // n
    image = image[:h_new * n, :w_new * n]  # Crop to make divisible by 5

    # Reshape into 5x5 blocks
    reshaped = image.reshape(h_new, n, w_new, n)

    # Compute the median for each 5x5 block
    binned = np.median(reshaped, axis=(1, 3))

    return binned

def disparity_to_3d(disparity, f, B, cx, cy, n):
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
    f = f / n
    cx = cx / n
    cy = cy / n

    # Get the image dimensions
    h, w = disparity.shape

    # Create a grid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Avoid division by zero by masking invalid disparity values
    valid_mask = disparity > 0

    # Compute depth (Z)
    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid_mask] = (f * B) / (disparity[valid_mask] * 0.125 / n)

    # Compute X and Y
    X = (x_coords - cx) * Z / f
    Y = (y_coords - cy) * Z / f

    # Stack X, Y, Z into an Nx3 array of 3D points
    points_3d = np.stack((Z[valid_mask], -X[valid_mask], -Y[valid_mask]), axis=-1)

    return points_3d

# obstacle (from point cloud)
# currently unused
def obs_callback(msg):
    #print(type(msg.points))
    #for p in msg.points:
    #    print(p.x, p.y, p.z)
    global latest_obs
    latest_obs = msg.points

# disparity to point cloud
# do disparity binning and convert to point cloud
# 
def disp_callback(img_msg):
    global latest_obs
    n=10
    binned = median_bin(np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width), n)
    latest_obs = disparity_to_3d(binned, 470.051, 0.0750492, 314.96, 229.359, n)
    header = Header()
    header.frame_id = "body"
    header.stamp = node.get_clock().now().to_msg()
    pc_pub.publish(point_cloud2.create_cloud_xyz32(header, latest_obs))

rclpy.init()
node = rclpy.create_node('obs_avd')

# Define a point in the "map" frame
# that is the target
point_in_map = PointStamped()
point_in_map.header.frame_id = "map"
point_in_map.header.stamp = node.get_clock().now().to_msg()
point_in_map.point.x = 3.0
point_in_map.point.y = 0.0
point_in_map.point.z = 0.5

latest_obs = None

# Create a TF2 buffer and listener
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
#obs_sub = node.create_subscription(PointCloud, "obstacles", obs_callback, 1)
pc_pub = node.create_publisher(PointCloud2, "obstacles", best_effort_qos)
disp_sub = node.create_subscription(Image, "oakd/disparity", disp_callback, qos_profile=best_effort_qos)
avd_pub = node.create_publisher(TwistStamped, "avoid_direction", 1)
# added: publish the destination
target_map_point_pub= node.create_publisher( PointStamped, 'target_map_point', 10)
# added Subscribe to the UAV's trajectory
# trajectory_subscription = node.create_subscription( PointStamped, 'uav_trajectory', trajectory_callback, 10)
# latest_trajectory_point = None

prv_yaw = None
prv_pitch = None
# experimenting: previous histogram
histogram_prev = np.zeros((360 // bin_sz, 180 // bin_sz)) # yaw x pitch

while rclpy.ok():
    try:
        rclpy.spin_once(node)
        # publish the target so we can see it!
        target_map_point_pub.publish(point_in_map)
        # logger.info(f'Published PointStamped in the map frame: {point_in_map}')
        if latest_obs is not None:
            try:
                # Lookup the transform from "map" to "body"
                transform = tf_buffer.lookup_transform(
                    "body",  # Target frame
                    "map",   # Source frame
                    rclpy.time.Time(),  # Use the latest available transform
                    timeout=rclpy.duration.Duration(seconds=0.0)
                    # timeout=rclpy.duration.Duration(seconds=0.0)
                )
                logger.info(f'got latest_obs: {latest_obs}')
            except Exception as e:
                logger.info(f'error did not get latest_obs')
                #print(e)
                pass
            else:
                # Transform the point
                point_in_body = do_transform_point(point_in_map, transform)
                # TODO add memory (previous histo)
                # best_yaw, best_pitch,histogram_prev = vfh_star_3d_pointcloud_target_direction(latest_obs,histogram_prev, np.array([point_in_body.point.x, point_in_body.point.y, point_in_body.point.z]), prv_yaw, prv_pitch, safety_distance=1.0, alpha=0.2, prv_weight=0.4)
                best_yaw, best_pitch = vfh_star_3d_pointcloud_target_direction(latest_obs, np.array([point_in_body.point.x, point_in_body.point.y, point_in_body.point.z]), prv_yaw, prv_pitch, safety_distance=1.0, alpha=0.2, prv_weight=0.4)
                prv_yaw = best_yaw
                prv_pitch = best_pitch

                # Convert spherical coordinates to a 3D vector.
                x = math.cos(best_pitch) * math.cos(best_yaw)
                y = math.cos(best_pitch) * math.sin(best_yaw)
                z = math.sin(best_pitch)

                # Normalize the vector.
                v = np.array([x, y, z])
                avd_dir = v / np.linalg.norm(v)

                avd_vel = avd_dir * 0.3
                m = TwistStamped()
                m.header.frame_id = "body"
                m.header.stamp = node.get_clock().now().to_msg()
                m.twist.linear.x = avd_vel[0]
                m.twist.linear.y = avd_vel[1]
                m.twist.linear.z = avd_vel[2]
                avd_pub.publish(m)

                latest_obs = None
    except KeyboardInterrupt:
        break
rclpy.try_shutdown()
print("bye")


