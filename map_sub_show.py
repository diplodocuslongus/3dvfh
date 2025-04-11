# Subscribe to the point cloud in the body frame (/body_cloud).
# Subscribe to the transformation from map to body via tf2_ros.TransformListener and tf2_ros.Buffer.
# Subscribe to the UAV's trajectory (/uav_trajectory).
# Publish a single point in the map frame (/map_point). For demonstration, we will transform a fixed point (defined in the body frame) to the map frame and publish it.
# use with uav_trajectory_pub.py and rviz config file tf_mapbody.rviz 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np

class MapPointPublisher(Node):
    def __init__(self):
        super().__init__('map_point_publisher')

        # Subscribe to the point cloud in the body frame
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            'body_cloud',
            self.cloud_callback,
            10
        )
        self.latest_cloud = None

        # Subscribe to the UAV's trajectory
        self.trajectory_subscription = self.create_subscription(
            PointStamped,
            'uav_trajectory',
            self.trajectory_callback,
            10
        )
        self.latest_trajectory_point = None

        # Create a tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for the point in the map frame
        self.map_point_publisher = self.create_publisher(
            PointStamped,
            'map_point',
            10
        )

        # Define a point in the body frame that we want to transform to the map frame
        self.body_fixed_point = np.array([1.0, 0.5, 0.0])

        # Publish the map point periodically
        self.timer = self.create_timer(1.0, self.publish_map_point)

    def cloud_callback(self, msg):
        self.latest_cloud = msg
        self.get_logger().info('Received point cloud')

    def trajectory_callback(self, msg):
        self.latest_trajectory_point = msg
        self.get_logger().info('Received trajectory point')

    def publish_map_point(self):
        if self.latest_cloud is None:
            self.get_logger().warn('No point cloud received yet.')
            return

        try:
            # Get the transform from the body frame to the map frame at the time of the point cloud
            transform = self.tf_buffer.lookup_transform(
                'map',
                'body',
                self.latest_cloud.header.stamp,
                rclpy.duration.Duration(seconds=1.0)  # Timeout if transform is not available
            )

            # Transform the fixed point from the body frame to the map frame
            map_point_msg = PointStamped()
            map_point_msg.header.stamp = self.get_clock().now().to_msg()
            map_point_msg.header.frame_id = 'map'
            map_point_msg.point.x = transform.transform.translation.x + self.body_fixed_point[0] * math.cos(self.get_yaw(transform)) - self.body_fixed_point[1] * math.sin(self.get_yaw(transform)) # Simple rotation for x
            map_point_msg.point.y = transform.transform.translation.y + self.body_fixed_point[0] * math.sin(self.get_yaw(transform)) + self.body_fixed_point[1] * math.cos(self.get_yaw(transform)) # Simple rotation for y
            map_point_msg.point.z = transform.transform.translation.z + self.body_fixed_point[2]

            self.map_point_publisher.publish(map_point_msg)
            self.get_logger().info(f'Published map point: ({map_point_msg.point.x}, {map_point_msg.point.y}, {map_point_msg.point.z})')

        except TransformException as ex:
            self.get_logger().info(f'Transform exception: {ex}')

    def get_yaw(self, transform):
        # Simple way to get yaw from quaternion (assuming no roll or pitch for simplicity here)
        q_x = transform.transform.rotation.x
        q_y = transform.transform.rotation.y
        q_z = transform.transform.rotation.z
        q_w = transform.transform.rotation.w
        siny_cosp = 2 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

def main(args=None):
    rclpy.init(args=args)
    node = MapPointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
