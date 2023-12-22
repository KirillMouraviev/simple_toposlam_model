import rospy
import os
import numpy as np
np.float = np.float64
import ros_numpy
from sensor_msgs.msg import PointCloud2

tests_dir = '/home/kirill/TopoSLAM/OpenPlaceRecognition/test_registration'

class Localizer():
    def __init__(self, graph, gt_map, publish=True):
        self.graph = graph
        self.gt_map = gt_map
        self.cloud = None
        self.x = None
        self.y = None
        self.theta = None
        self.stamp = None
        self.localized_x = None
        self.localized_y = None
        self.localized_theta = None
        self.rel_poses = None
        self.dists = None
        self.cnt = 0
        if not os.path.exists(tests_dir):
            os.mkdir(tests_dir)
        self.publish = publish
        self.cand_cloud_publisher = rospy.Publisher('/candidate_cloud', PointCloud2, latch=True, queue_size=100)

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

    def get_rel_pose(self, x, y, theta, x2, y2):
        angle = self.normalize(np.arctan2(y2 - y, x2 - x) - theta)
        rel_x, rel_y = self.rotate(x2 - x, y2 - y, angle)
        return rel_x, rel_y

    def save_reg_test_data(self, vertex_ids, transforms, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savetxt(os.path.join(save_dir, 'ref_cloud.txt'), self.cloud)
        tf_data = []
        gt_pose_data = [[self.x, self.y, self.theta]]
        for idx, tf in zip(vertex_ids, transforms):
            if idx >= 0:
                print('idx:', idx)
                x, y, theta, cloud = self.graph.vertices[idx]
                print('GT x, y, theta:', x, y, theta)
                np.savetxt(os.path.join(save_dir, 'cand_cloud_{}.txt'.format(idx)), cloud)
                tf_data.append([idx] + list(tf))
                gt_pose_data.append([x, y, theta])
        print('TF data:', tf_data)
        np.savetxt(os.path.join(save_dir, 'gt_poses.txt'), np.array(gt_pose_data))
        np.savetxt(os.path.join(save_dir, 'transforms.txt'), np.array(tf_data))

    def localize(self, event=None):
        x = self.x
        y = self.y
        theta = self.theta
        vertex_ids = []
        rel_poses = []
        dists = []
        if self.cloud is not None:
            vertex_ids_pr, transforms = self.graph.get_k_most_similar(self.cloud, k=5)
            save_dir = os.path.join(tests_dir, 'test_{}'.format(self.cnt))
            self.cnt += 1
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_reg_test_data(vertex_ids_pr, transforms, save_dir)
        else:
            vertex_ids_pr = []
        vertex_ids_pr = [i for i in vertex_ids_pr if i >= 0]
        #for i, v in enumerate(self.graph.vertices):
        for i in vertex_ids_pr:
            v = self.graph.vertices[i]
            dist = np.sqrt((x - v[0]) ** 2 + (y - v[1]) ** 2)
            if self.gt_map.in_sight(x, y, v[0], v[1]):
                vertex_ids.append(i)
                rel_poses.append(self.get_rel_pose(x, y, theta, v[0], v[1]))
                dists.append(dist)
        ids = list(range(len(dists)))
        ids.sort(key=lambda i: dists[i])
        vertex_ids = [vertex_ids[i] for i in ids]
        rel_poses = [rel_poses[i] for i in ids]
        dists = [dists[i] for i in ids]
        if len(vertex_ids) > 0:
            i = vertex_ids[0]
            cloud = self.graph.vertices[i][3]
            cloud_with_fields = np.zeros((cloud.shape[0]), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('r', np.uint8),
                ('g', np.uint8),
                ('b', np.uint8)])
            cloud_with_fields['x'] = cloud[:, 0]
            cloud_with_fields['y'] = cloud[:, 1]
            cloud_with_fields['z'] = cloud[:, 2]
            cloud_with_fields['r'] = cloud[:, 3]
            cloud_with_fields['g'] = cloud[:, 4]
            cloud_with_fields['b'] = cloud[:, 5]
            cloud_with_fields = ros_numpy.point_cloud2.merge_rgb_fields(cloud_with_fields)
            cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_with_fields)
            if self.stamp is None:
                cloud_msg.header.stamp = rospy.Time.now()
            else:
                cloud_msg.header.stamp = self.stamp
            cloud_msg.header.frame_id = 'base_link'
            self.cand_cloud_publisher.publish(cloud_msg)
        self.localized_x = self.x
        self.localized_y = self.y
        self.localized_theta = self.theta
        self.localized_cloud = self.cloud
        return vertex_ids, rel_poses, dists