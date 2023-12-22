#!/usr/bin/env python

import rospy
import numpy as np
np.float = np.float64
import ros_numpy
import os
import sys
import tf
from localization import Localizer
from gt_map import GTMap
#from mapping import Mapper
from topo_graph import TopologicalGraph
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point
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
        self.localization_subscriber = rospy.Subscriber('/localized_nodes', Float32MultiArray, self.localization_callback)
        self.gt_map_publisher = rospy.Publisher('/habitat/gt_map', OccupancyGrid, latch=True, queue_size=100)
        self.task_publisher = rospy.Publisher('/task', Float32MultiArray, latch=True, queue_size=100)
        self.last_vertex_publisher = rospy.Publisher('/last_vertex', Marker, latch=True, queue_size=100)
        self.localization_results_publisher = rospy.Publisher('/localized_vertices', Marker, latch=True, queue_size=100)
        self.localization_time = 0
        self.localization_results = ([], [], [])
        rospy.Timer(rospy.Duration(1.0), self.localizer.localize)
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

    def get_xyz_coords_from_msg(self, msg):
        points_numpify = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        points_numpify = ros_numpy.point_cloud2.split_rgb_field(points_numpify)
        points_x = np.array([x[2] for x in points_numpify])[:, np.newaxis]
        points_y = np.array([-x[0] for x in points_numpify])[:, np.newaxis]
        points_z = np.array([-x[1] for x in points_numpify])[:, np.newaxis]
        points_r = np.array([x[3] for x in points_numpify])[:, np.newaxis]
        points_g = np.array([x[4] for x in points_numpify])[:, np.newaxis]
        points_b = np.array([x[5] for x in points_numpify])[:, np.newaxis]
        points_xyz = np.concatenate([points_x, points_y, points_z, points_r, points_g, points_b], axis=1)
        return points_xyz

    def transform_pcd(self, points, x, y, theta):
        points_transformed = points.copy()
        points_transformed[:, 0] = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
        points_transformed[:, 1] = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
        points_transformed[:, 0] -= x
        points_transformed[:, 1] -= y
        return points_transformed

    def get_occupancy_grid(self, points_xyz):
        points_xyz = np.clip(points_xyz, -8, 8)
        resolution = 0.1
        radius = 18
        points_ij = np.round(points_xyz[:, :2] / resolution).astype(int) + [int(radius / resolution), int(radius / resolution)]
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

    def publish_localization_results(self, vertex_ids):
        vertices_marker = Marker()
        #vertices_marker = ns = 'points_and_lines'
        vertices_marker.type = Marker.POINTS
        vertices_marker.id = 0
        vertices_marker.header.frame_id = 'map'
        vertices_marker.header.stamp = rospy.Time.now()
        vertices_marker.scale.x = 0.2
        vertices_marker.scale.y = 0.2
        vertices_marker.scale.z = 0.2
        vertices_marker.color.r = 1
        vertices_marker.color.g = 1
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        localized_vertices = [self.graph.vertices[i] for i in vertex_ids]
        for x, y, theta, cloud in localized_vertices:
            vertices_marker.points.append(Point(x, y, 0.1))
        self.localization_results_publisher.publish(vertices_marker)
        
    def localization_callback(self, msg):
        self.localization_time = rospy.Time.now().to_sec()
        #self.publish_cur_cloud()
        n = msg.layout.dim[0].size // 3
        vertex_ids = [int(x) for x in msg.data[:n]]
        thetas = msg.data[n:2 * n]
        dists = msg.data[2 * n:3 * n]
        self.localization_results = (vertex_ids, thetas, dists)
        print('Localized in vertices', vertex_ids)
        self.publish_localization_results(vertex_ids)
        x = self.localizer.localized_x
        y = self.localizer.localized_y
        theta = self.localizer.localized_theta
        cloud = self.localizer.localized_cloud
        if len(vertex_ids) == 0:
            return
        if self.last_vertex is None:
            self.last_vertex_id = vertex_ids[0]
            self.last_vertex = self.graph.get_vertex(vertex_ids[0])
        else:
            found_loop_closure = False
            for i in range(len(vertex_ids)):
                for j in range(len(vertex_ids)):
                    u = vertex_ids[i]
                    v = vertex_ids[j]
                    path, path_len = self.graph.get_path_with_length(u, v)
                    dst_through_cur = dists[i] + dists[j]
                    if path_len > 5 and path_len > 2 * dst_through_cur:
                        print('u:', self.graph.get_vertex(u)[0], self.graph.get_vertex(u)[1])
                        print('v:', self.graph.get_vertex(v)[0], self.graph.get_vertex(v)[1])
                        print('Path in graph:', path_len)
                        print('Path through cur:', dst_through_cur)
                        found_loop_closure = True
                        break
                if found_loop_closure:
                    break
            if found_loop_closure:
                print('Found loop closure. Add new vertex to close loop')
                new_vertex_id = self.graph.add_vertex(x, y, theta, cloud)
                self.last_vertex_id = new_vertex_id
                self.last_vertex = self.graph.get_vertex(new_vertex_id)
                for i in range(len(vertex_ids)):
                    self.graph.add_edge(new_vertex_id, vertex_ids[i], thetas[i], dists[i])
            else:
                found_proper_vertex = False
                for v in vertex_ids:
                    vx, vy, vtheta, vcloud = self.graph.get_vertex(v)
                    if (v == self.last_vertex_id or self.graph.has_edge(v, self.last_vertex_id)) and \
                        self.get_iou(vx, vy, vtheta, vcloud, self.last_vertex) > self.iou_threshold2:
                        print('Change vertex to ({}, {})'.format(self.graph.get_vertex(v)[0], self.graph.get_vertex(v)[1]))
                        found_proper_vertex = True
                        self.last_vertex_id = v
                        self.last_vertex = self.graph.get_vertex(v)
                        break
                if not found_proper_vertex:
                    print('No proper vertex to change. Add new vertex')
                    new_vertex_id = self.graph.add_vertex(x, y, theta, cloud)
                    self.last_vertex_id = new_vertex_id
                    self.last_vertex = self.graph.get_vertex(new_vertex_id)
                    for i in range(len(vertex_ids)):
                        self.graph.add_edge(new_vertex_id, vertex_ids[i], thetas[i], dists[i])

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
        if self.last_vertex is None:
            print('Add new vertex at start')
            new_vertex_id = self.graph.add_vertex(x, y, theta, cur_cloud)
            self.last_vertex_id = new_vertex_id
            self.last_vertex = self.graph.get_vertex(new_vertex_id)
        in_sight = self.is_in_sight(x, y, self.last_vertex[0], self.last_vertex[1])
        iou = self.get_iou(x, y, theta, cur_cloud, self.last_vertex)
        #print('In sight:', in_sight)
        #print('IoU:', iou)
        if in_sight is None:
            print('Failed to check straight-line visibility!')
            return
        if iou < self.iou_threshold or not in_sight:
            if not in_sight:
                print('Out of visibility')
            else:
                print('Low IoU')
            if rospy.Time.now().to_sec() - self.localization_time < 1:
                vertex_ids, thetas, dists = self.localization_results
                found_proper_vertex = False
                for v in vertex_ids:
                    iou = self.get_iou(x, y, theta, cur_cloud, self.graph.get_vertex(v))
                    print('IoU between ({}, {}) and ({}, {}) is {}'.format(x, y, self.graph.get_vertex(v)[0], self.graph.get_vertex(v)[1], iou))
                    print('Has edge:', self.graph.has_edge(self.last_vertex_id, v))
                    if self.graph.has_edge(self.last_vertex_id, v) and iou > self.iou_threshold2:
                        found_proper_vertex = True
                        self.last_vertex_id = v
                        self.last_vertex = self.graph.get_vertex(v)
                        print('Change to vertex ({}, {})'.format(self.last_vertex[0], self.last_vertex[1]))
                        break
                if not found_proper_vertex:
                    print('No proper vertex to change. Add new vertex')
                    new_vertex_id = self.graph.add_vertex(self.prev_x, self.prev_y, self.prev_theta, self.prev_cloud)
                    new_vertex = self.graph.get_vertex(new_vertex_id)
                    rel_x, rel_y, rel_theta = self.get_rel_pose(self.last_vertex[0], self.last_vertex[1], self.last_vertex[2], 
                                                                new_vertex[0], new_vertex[1], new_vertex[2])
                    theta = np.arctan2(rel_y, rel_x)
                    dst = np.sqrt(rel_x ** 2 + rel_y ** 2)
                    self.graph.add_edge(new_vertex_id, self.last_vertex_id, theta, dst)
                    for v, theta, dst in zip(vertex_ids, thetas, dists):
                        self.graph.add_edge(new_vertex_id, v, theta, dst)
                    self.last_vertex_id = new_vertex_id
                    self.last_vertex = new_vertex
            else:
                new_vertex_id = self.graph.add_vertex(self.prev_x, self.prev_y, self.prev_theta, self.prev_cloud)
                new_vertex = self.graph.get_vertex(new_vertex_id)
                rel_x, rel_y, rel_theta = self.get_rel_pose(self.last_vertex[0], self.last_vertex[1], self.last_vertex[2], 
                                                            new_vertex[0], new_vertex[1], new_vertex[2])
                theta = np.arctan2(rel_y, rel_x)
                dst = np.sqrt(rel_x ** 2 + rel_y ** 2)
                self.graph.add_edge(new_vertex_id, self.last_vertex_id, theta, dst)
                self.last_vertex_id = new_vertex_id
                self.last_vertex = self.graph.get_vertex(new_vertex_id)
        self.graph.publish_graph()
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
        cur_x, cur_y, cur_theta = cur_pose
        cur_cloud = self.get_xyz_coords_from_msg(msg)
        self.localizer.x = cur_x
        self.localizer.y = cur_y
        self.localizer.theta = cur_theta
        self.localizer.cloud = cur_cloud
        self.localizer.stamp = msg.header.stamp
        self.update_by_iou(cur_x, cur_y, cur_theta, cur_cloud, msg.header.stamp)

    def save_graph(self, save_dir='src/simple_toposlam_model/grids'):
        self.graph.save_to_json(self.path_to_save_json)

    def run(self):
        rospy.spin()


toposlam_model = TopoSLAMModel()
toposlam_model.run()
toposlam_model.save_graph()