#!/usr/bin/env python

import rospy
import ros_numpy
import os
import tf
import numpy as np
from localization import Localizer
from gt_map import GTMap
#from mapping import Mapper
from topo_graph import TopologicalGraph
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker

rospy.init_node('simple_toposlam_model')

class TopoSLAMModel():
    def __init__(self):
        self.graph = TopologicalGraph()
        scene_name = rospy.get_param('~scene_name')
        self.path_to_save_json = rospy.get_param('~path_to_save_json')
        self.iou_threshold = rospy.get_param('~iou_threshold', 0.2)
        self.iou_threshold2 = 0.4
        gt_map_dir = '/home/kirill/TopoSLAM/GT/{}'.format(scene_name)
        gt_map_filename = [f for f in os.listdir(gt_map_dir) if f.startswith('map_cropped_')][0]
        gt_map = GTMap(os.path.join(gt_map_dir, gt_map_filename))
        self.localizer = Localizer(self.graph, gt_map)
        self.gt_map = gt_map
        self.pcd_subscriber = rospy.Subscriber('/habitat/points', PointCloud2, self.pcd_callback)
        self.pose_subscriber = rospy.Subscriber('/true_pose', PoseStamped, self.pose_callback)
        self.in_sight_response_subscriber = rospy.Subscriber('/response', Float32MultiArray, self.in_sight_callback)
        self.gt_map_publisher = rospy.Publisher('/habitat/gt_map', OccupancyGrid, latch=True, queue_size=100)
        self.task_publisher = rospy.Publisher('/task', Float32MultiArray, latch=True, queue_size=100)
        self.last_vertex_publisher = rospy.Publisher('/last_vertex', Marker, latch=True, queue_size=100)
        self.last_vertex = None
        self.last_vertex_id = None
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.clouds = {}
        self.prev_cloud = None
        self.in_sight_response = None
        self.poses = []
        self.pose_pairs = []
        self.cur_grids = []
        self.cur_grids_transformed = []
        self.ref_grids = []

    def get_xy_coords_from_msg(self, msg):
        points_numpify = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points_numpify = ros_numpy.point_cloud2.split_rgb_field(points_numpify)
        points_x = np.array([x[2] for x in points_numpify])[:, np.newaxis]
        points_y = np.array([-x[0] for x in points_numpify])[:, np.newaxis]
        points_xy = np.concatenate([points_x, points_y], axis=1)
        return points_xy

    def transform_pcd(self, points, x, y, theta):
        points_transformed = points.copy()
        points_transformed[:, 0] = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
        points_transformed[:, 1] = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
        points_transformed[:, 0] -= x
        points_transformed[:, 1] -= y
        return points_transformed

    def get_occupancy_grid(self, points_xy):
        points_xy = np.clip(points_xy, -8, 8)
        resolution = 0.1
        radius = 18
        points_ij = np.round(points_xy / resolution).astype(int) + [int(radius / resolution), int(radius / resolution)]
        grid = np.zeros((int(2 * radius / resolution), int(2 * radius / resolution)), dtype=np.uint8)
        grid[points_ij[:, 0], points_ij[:, 1]] = 1
        return grid

    def publish_gt_map(self):
        gt_map_msg = OccupancyGrid()
        gt_map_msg.header.stamp = rospy.Time.now()
        gt_map_msg.header.frame_id = 'map'
        gt_map_msg.info.resolution = 0.05
        gt_map_msg.info.width = self.gt_map.gt_map.shape[1]
        gt_map_msg.info.height = self.gt_map.gt_map.shape[0]
        gt_map_msg.info.origin.position.x = -24 + self.gt_map.start_j / 20
        gt_map_msg.info.origin.position.y = -24 + self.gt_map.start_i / 20
        gt_map_msg.info.origin.orientation.x = 0
        gt_map_msg.info.origin.orientation.y = 0
        gt_map_msg.info.origin.orientation.z = 0
        gt_map_msg.info.origin.orientation.w = 1
        gt_map_ravel = self.gt_map.gt_map.ravel()
        gt_map_data = self.gt_map.gt_map.ravel().astype(np.int8)
        gt_map_data[gt_map_ravel == 0] = 100
        gt_map_data[gt_map_ravel == 127] = -1
        gt_map_data[gt_map_ravel == 255] = 0
        gt_map_msg.data = list(gt_map_data)
        self.gt_map_publisher.publish(gt_map_msg)

    def normalize(self, angle):
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def rotate(self, x, y, angle):
        x_new = x * np.cos(angle) + y * np.sin(angle)
        y_new = -x * np.sin(angle) + y * np.cos(angle)
        return x_new, y_new

    def get_rel_pose(self, x, y, theta, x2, y2, theta2):
        #angle = self.normalize(np.arctan2(y2 - y, x2 - x) - theta)
        rel_x, rel_y = self.rotate(x2 - x, y2 - y, theta2)
        return rel_x, rel_y, theta2 - theta

    def get_iou(self, x, y, theta, cloud, vertex):
        v_x, v_y, v_theta, v_cloud = vertex
        rel_x, rel_y, rel_theta = self.get_rel_pose(x, y, theta, v_x, v_y, v_theta)
        cur_cloud_transformed = self.transform_pcd(cloud, rel_x, rel_y, rel_theta)
        cur_grid_transformed = self.get_occupancy_grid(cur_cloud_transformed)
        cur_grid = self.get_occupancy_grid(cloud)
        v_grid = self.get_occupancy_grid(v_cloud)
        intersection = np.sum(v_grid * cur_grid_transformed)
        union = np.sum(v_grid | cur_grid_transformed)
        return intersection / union

    def pose_callback(self, msg):
        x, y = msg.pose.position.x, msg.pose.position.y
        orientation = msg.pose.orientation
        _, __, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.poses.append([msg.header.stamp.to_sec(), x, y, theta])

    def get_sync_pose(self, timestamp):
        if len(self.poses) == 0:
            print('No pose available!')
            return None
        i = 0
        while i < len(self.poses) and self.poses[i][0] < timestamp:
            i += 1
        if i == 0:
            if self.poses[0][0] - timestamp > 0.2:
                print('No sync pose available!')
                return None
            return self.poses[0][1:]
        if i == len(self.poses):
            print('No sync pose available!')
            return None
        alpha = (timestamp - self.poses[i - 1][0]) / (self.poses[i][0] - self.poses[i - 1][0])
        pose_left = np.array(self.poses[i - 1][1:])
        pose_right = np.array(self.poses[i][1:])
        return alpha * pose_right + (1 - alpha) * pose_left

    def in_sight_callback(self, msg):
        self.in_sight_response = msg.data

    def is_equal(self, a, b):
        for i in range(4):
            if abs(a[i] - b[i]) > 1e-5:
                return False
        return True

    def is_in_sight(self, x1, y1, x2, y2):
        dst = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dst > 5:
            return False
        self.publish_gt_map()
        task_msg = Float32MultiArray()
        task_msg.layout.dim.append(MultiArrayDimension())
        task_msg.layout.dim[0].label = "width"
        task_msg.layout.dim[0].size = 4
        task_msg.layout.dim[0].stride = 4
        task_msg.layout.data_offset = 0
        task_msg.data = [x1, y1, x2, y2]
        self.task_publisher.publish(task_msg)
        start_time = rospy.Time.now().to_sec()
        while self.in_sight_response is None or not self.is_equal(self.in_sight_response[:4], task_msg.data):
            rospy.sleep(1e-3)
            if rospy.Time.now().to_sec() - start_time > 0.5:
                print('Waiting for response timed out!')
                return None
        return bool(self.in_sight_response[4])

    def publish_last_vertex(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = 'map'
        marker_msg.type = Marker.SPHERE
        marker_msg.pose.position.x = self.last_vertex[0]
        marker_msg.pose.position.y = self.last_vertex[1]
        marker_msg.color.r = 0
        marker_msg.color.g = 1
        marker_msg.color.b = 0
        marker_msg.color.a = 1
        marker_msg.scale.x = 0.5
        marker_msg.scale.y = 0.5
        marker_msg.scale.z = 0.5
        self.last_vertex_publisher.publish(marker_msg)

    def update_by_iou(self, x, y, theta, cur_cloud, timestamp):
        #print('x y theta:', x, y, theta)
        t1 = rospy.Time.now().to_sec()
        if cur_cloud is None:
            print('No point cloud received!')
            return
        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            self.prev_theta = theta
            self.prev_cloud = cur_cloud
        in_sight = None
        iou = None
        if self.last_vertex is not None:
            in_sight = self.is_in_sight(x, y, self.last_vertex[0], self.last_vertex[1])
            iou = self.get_iou(x, y, theta, cur_cloud, self.last_vertex)
            if in_sight is None:
                print('Failed to check straight-line visibility!')
                return
        #print('In sight:', in_sight)
        #print('IoU:', iou)
        t2 = rospy.Time.now().to_sec()
        if self.last_vertex is None or iou < self.iou_threshold or not in_sight:
            if self.last_vertex is not None:
                print('Last vertex:', self.last_vertex[0], self.last_vertex[1])
            vertex_ids, rel_poses, dists = self.localizer.localize(x, y, theta)
            t3 = rospy.Time.now().to_sec()
            #print('Localization time:', t3 - t2)
            if len(rel_poses) == 0:
                print('Localization failed. Add new vertex')
                prev_vertex_ids, prev_rel_poses, prev_dists = self.localizer.localize(self.prev_x, self.prev_y, self.prev_theta)
                new_vertex_id = self.graph.add_vertex(self.prev_x, self.prev_y, self.prev_theta, self.prev_cloud)
                self.last_vertex = self.graph.get_vertex(new_vertex_id)
                self.last_vertex_id = new_vertex_id
                for vertex_id in prev_vertex_ids:
                    self.graph.add_edge(new_vertex_id, vertex_id)
                self.graph.publish_graph()
                t4 = rospy.Time.now().to_sec()
                #print('Vertex addition time:', t4 - t3)
            else:
                found_proper_vertex = False
                for v in vertex_ids:
                    if v == self.last_vertex_id:
                        continue
                    neighbour_vertex = self.graph.get_vertex(v)
                    print('Vertex:', neighbour_vertex[0], neighbour_vertex[1])
                    iou = self.get_iou(x, y, theta, cur_cloud, neighbour_vertex)
                    print('IoU:', iou)
                    print('Has edge:', self.graph.has_edge(self.last_vertex_id, v))
                    if iou >= self.iou_threshold and self.graph.has_edge(self.last_vertex_id, v):
                        print('Change to vertex', neighbour_vertex[0], neighbour_vertex[1])
                        self.last_vertex = neighbour_vertex
                        self.last_vertex_id = v
                        found_proper_vertex = True
                        break
                if not found_proper_vertex:
                    print('No proper vertex to change. Add new vertex')
                    new_vertex_id = self.graph.add_vertex(self.prev_x, self.prev_y, self.prev_theta, self.prev_cloud)
                    self.last_vertex = self.graph.get_vertex(new_vertex_id)
                    self.last_vertex_id = new_vertex_id
                    prev_vertex_ids, prev_rel_poses, prev_dists = self.localizer.localize(self.prev_x, self.prev_y, self.prev_theta)
                    for vertex_id in prev_vertex_ids:
                        self.graph.add_edge(new_vertex_id, vertex_id)
                    self.graph.publish_graph()
                t4 = rospy.Time.now().to_sec()
                #print('Vertex addition time 2:', t4 - t3)
        self.publish_last_vertex()
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta
        self.prev_cloud = cur_cloud

    def pcd_callback(self, msg):
        cur_pose = self.get_sync_pose(msg.header.stamp.to_sec())
        start_time = rospy.Time.now().to_sec()
        while cur_pose is None:
            cur_pose = self.get_sync_pose(msg.header.stamp.to_sec())
            rospy.sleep(1e-2)
            if rospy.Time.now().to_sec() - start_time > 0.5:
                print('Waiting for sync pose timed out!')
                return
        #print('Time of waiting for sync pose:', rospy.Time.now().to_sec() - start_time)
        cur_x, cur_y, cur_theta = cur_pose
        t2 = rospy.Time.now().to_sec()
        cur_cloud = self.get_xy_coords_from_msg(msg)
        t3 = rospy.Time.now().to_sec()
        self.localizer.x = cur_x
        self.localizer.y = cur_y
        self.localizer.theta = cur_theta
        self.localizer.cloud = cur_cloud
        #print('Time of extracting point cloud:', t3 - t2)
        self.update_by_iou(cur_x, cur_y, cur_theta, cur_cloud, msg.header.stamp)
        t4 = rospy.Time.now().to_sec()
        #print('Time of updating graph:', t4 - t3)

    def save_graph(self, save_dir='src/simple_toposlam_model/grids'):
        self.graph.save_to_json(self.path_to_save_json)

    def run(self):
        rospy.spin()


toposlam_model = TopoSLAMModel()
toposlam_model.run()
toposlam_model.save_graph()