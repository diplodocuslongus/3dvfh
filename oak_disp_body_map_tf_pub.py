# added: transform body to map
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import depthai as dai
import cv2
import numpy as np
from enum import IntEnum

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

class UAVPublisher(Node):
    def __init__(self):
        super().__init__('uav_publisher')

        # Define the start and end points for the UAV's trajectory
        self.start_position = np.array([1.0, -0.5, 0.2])
        self.end_position = np.array([5.0, 2.0, 1.5])
        # self.total_duration = 10.0  # seconds for the animation
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        # self.animation_rate = 30.0  # Hz
        self.fps = 30 

        # Publisher for the UAV's trajectory points
        self.trajectory_publisher = self.create_publisher(
            PointStamped, 'uav_trajectory', 10
        )

        # Transform broadcaster to publish the body frame's pose in the map frame
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / self.fps, self.animation_step)
        # self.timer = self.create_timer(1.0 / self.fps, self.publish_body_transform)

        self.get_logger().info('UAV publisher started.')


    def publish_trajectory_point(self, position):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'map'
        point_msg.point.x = position[0]
        point_msg.point.y = position[1]
        point_msg.point.z = position[2]
        self.trajectory_publisher.publish(point_msg)

    def publish_body_transform(self, current_position, roll=0.0, pitch=0.0, yaw=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'body'
        t.transform.translation.x = current_position[0]
        t.transform.translation.y = current_position[1]
        t.transform.translation.z = current_position[2]

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy

        t.transform.rotation.x = q_x
        t.transform.rotation.y = q_y
        t.transform.rotation.z = q_z
        t.transform.rotation.w = q_w

        self.tf_broadcaster.sendTransform(t)

    def animation_step(self):
        current_position = self.start_position 
        self.publish_body_transform(current_position)
        # self.cloud_publisher.publish(self.generate_point_cloud())
        self.publish_trajectory_point(current_position)


class UAVPublisherAnim(Node):
    def __init__(self):
        super().__init__('uav_publisher')

        # Define the start and end points for the UAV's trajectory
        self.start_position = np.array([1.0, -0.5, 0.2])
        self.end_position = np.array([5.0, 2.0, 1.5])
        self.total_duration = 10.0  # seconds for the animation
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.animation_rate = 30.0  # Hz

        # Publisher for the point cloud in the body frame
        self.cloud_publisher = self.create_publisher(
            PointCloud2, 'body_cloud', 10
        )

        # Publisher for the UAV's trajectory points
        self.trajectory_publisher = self.create_publisher(
            PointStamped, 'uav_trajectory', 10
        )

        # Transform broadcaster to publish the body frame's pose in the map frame
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / self.animation_rate, self.animation_step)

        self.get_logger().info('UAV publisher started.')

    def generate_point_cloud(self):
        points = np.random.rand(100, 3).astype(np.float32) - 0.5
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'body'
        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg

    def publish_trajectory_point(self, position):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'map'
        point_msg.point.x = position[0]
        point_msg.point.y = position[1]
        point_msg.point.z = position[2]
        self.trajectory_publisher.publish(point_msg)

    def publish_body_transform(self, current_position, roll=0.0, pitch=0.0, yaw=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'body'
        t.transform.translation.x = current_position[0]
        t.transform.translation.y = current_position[1]
        t.transform.translation.z = current_position[2]

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy

        t.transform.rotation.x = q_x
        t.transform.rotation.y = q_y
        t.transform.rotation.z = q_z
        t.transform.rotation.w = q_w

        self.tf_broadcaster.sendTransform(t)

    def animation_step(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        if elapsed_time < self.total_duration:
            fraction = elapsed_time / self.total_duration
            current_position = (1 - fraction) * self.start_position + fraction * self.end_position
            self.publish_body_transform(current_position, yaw=fraction * 2 * math.pi)
            self.cloud_publisher.publish(self.generate_point_cloud())
            self.publish_trajectory_point(current_position)
        else:
            self.destroy_timer(self.timer)
            self.get_logger().info('UAV animation finished.')

class OakDDisparityPublisherNoBridge(Node):
    class ImageType(IntEnum):
        RAW = 0
        GRAY8 = 1
        RGB888 = 2
        BGR888 = 3

    def __init__(self):
        super().__init__('oakd_disparity_publisher')

        # self.start_position = np.array([1.0, -0.5, 0.2])
        # self.publisher_ = self.create_publisher(Image, 'disparity', 10)
        self.publisher_ = self.create_publisher(Image, 'oakd/disparity', 10)
        self.declare_parameter('mono_resolution', '400p')
        self.declare_parameter('fps', 30)
        self.declare_parameter('confidence_threshold', 200)
        self.subpixelon=True
        self.bin_sz = (8, 8)
        if self.subpixelon:
            self.img_encoding = "mono16"
            self.img_step_factor = 2
        else:
            self.img_encoding = "mono8"
            self.img_step_factor = 2

        self.mono_resolution = self.get_parameter('mono_resolution').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().integer_value

        self.pipeline = dai.Pipeline()
        self.create_pipeline()
        self.device = dai.Device(self.pipeline)

        self.disparity_queue = self.device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

        self.timer = self.create_timer(1.0 / self.fps, self.publish_disparity)
        self.get_logger().info('OAK-D disparity publisher started')

    # depthai camera pipeline
    def create_pipeline(self):
        left = self.pipeline.create(dai.node.MonoCamera)
        right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        xout_disparity = self.pipeline.create(dai.node.XLinkOut)

        if self.mono_resolution == '400p':
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        elif self.mono_resolution == '480p':
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        elif self.mono_resolution == '720p':
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        elif self.mono_resolution == '800p':
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        elif self.mono_resolution == '1200p':
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_1200_P)
        else:
            self.get_logger().warn(f"Invalid mono_resolution: {self.mono_resolution}. Defaulting to 400p.")
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo.setConfidenceThreshold(self.confidence_threshold)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(self.subpixelon)

        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.disparity.link(xout_disparity.input)

        xout_disparity.setStreamName("disparity")

    def bin_dispmap_numpy(self,disparity_map, bin_size, method='mean'):
        height, width = disparity_map.shape
        bin_h, bin_w = bin_size

        # Calculate the new dimensions
        new_height = height // bin_h
        new_width = width // bin_w

        # Reshape the array to group pixels into bins
        reshaped = disparity_map[:new_height * bin_h, :new_width * bin_w].reshape(
            new_height, bin_h, new_width, bin_w
        )

        # Calculate the mean (or other statistic) along the bin axes (axis=1 and axis=3)
        if method == 'mean':
            binned_disparity = np.mean(reshaped, axis=(1, 3), dtype=np.float32)
        elif method == 'median':
            binned_disparity = np.median(reshaped, axis=(1, 3)).astype(np.float32)
        elif method == 'max':
            binned_disparity = np.max(reshaped, axis=(1, 3)).astype(np.float32)
        elif method == 'min':
            binned_disparity = np.min(reshaped, axis=(1, 3)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return binned_disparity

    def publish_disparity(self):
        in_disparity = self.disparity_queue.tryGet()

        # TODO review naming!
        if in_disparity is not None:
            disparity_frame = in_disparity.getFrame()
            # disparity = self.bin_dispmap_numpy(disparity_frame, self.bin_sz,method='mean')
            normalized_disparity = disparity_frame
            # normalized_disparity = cv2.normalize(disparity_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)

            img_msg = Image()
            img_msg.header = Header()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            # img_msg.height = colored_disparity.shape[0]
            # img_msg.width = colored_disparity.shape[1]
            img_msg.height = normalized_disparity.shape[0]
            img_msg.width = normalized_disparity.shape[1]
            # img_msg.encoding = "mono8" #i mono8 for opencv normalized image "bgr8" for color converted
            img_msg.encoding = self.img_encoding #"mono16" #i mono8 for opencv normalized image "bgr8" for color converted
            img_msg.is_bigendian = 0
            # img_msg.step = img_msg.width 
            img_msg.step = img_msg.width * self.img_step_factor 
            # img_msg.step = img_msg.width * 3  # bytes per row (BGR8 has 3 channels)
            img_msg.data = normalized_disparity.tobytes()
            # img_msg.data = normalized_disparity.ravel()
            # self.get_logger().info(f'HxW = {img_msg.height},{img_msg.width}, encodingg={img_msg.encoding},step {img_msg.step}') #,datasz={img_msg.data}')
            # expected_size = img_msg.height * img_msg.step
            # self.get_logger().info(f'Expected data size {expected_size}')

            try:
                self.publisher_.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f'Error publishing disparity image: {e}')

def main(args=None):
    rclpy.init(args=args)
    oakd_disparity_publisher = OakDDisparityPublisherNoBridge()
    bodytf_publisher = UAVPublisherAnim()
    # bodytf_publisher = UAVPublisherAnim()
    rclpy.spin(oakd_disparity_publisher)
    rclpy.spin(bodytf_publisher)
    oakd_disparity_publisher.destroy_node()
    bodytf_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
