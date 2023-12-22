import json
import rospy
import heapq
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from collections import deque
from typing import Dict
from torch import Tensor

import sys
sys.path.append('/home/kirill/TopoSLAM/OpenPlaceRecognition/src')
import faiss
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from opr.models.place_recognition import MinkLoc3D
from opr.pipelines.registration import PointcloudRegistrationPipeline
import MinkowskiEngine as ME


class TopologicalGraph():
    def __init__(self):
        self.vertices = []
        self.adj_lists = []
        self.pub = rospy.Publisher('topological_map', MarkerArray, latch=True, queue_size=100)

        WEIGHTS_PATH = "/home/kirill/TopoSLAM/OpenPlaceRecognition/weights/place_recognition/minkloc3d_nclt.pth"
        REGISTRATION_MODEL_CONFIG_PATH = "/home/kirill/TopoSLAM/OpenPlaceRecognition/configs/model/registration/geotransformer_kitti.yaml"
        REGISTRATION_WEIGHTS_PATH = "/home/kirill/TopoSLAM/OpenPlaceRecognition/weights/registration/geotransformer_kitti.pth"
        self.model = MinkLoc3D()
        self.model.load_state_dict(torch.load(WEIGHTS_PATH))
        self.model = self.model.to("cuda")
        self.model.eval()
        self.registration_model = instantiate(OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH))
        self.registration_model.load_state_dict(torch.load(REGISTRATION_WEIGHTS_PATH))
        self.registration_pipeline = PointcloudRegistrationPipeline(
                                        model=self.registration_model,
                                        model_weights_path=REGISTRATION_WEIGHTS_PATH,
                                        device="cuda",  # the GeoTransformer currently only supports CUDA
                                        voxel_downsample_size=0.3,  # recommended for geotransformer_kitti configuration
                                    )
        self.index = faiss.IndexFlatL2(256)
        self._pointcloud_quantization_size = 0.5
        self.device = torch.device('cuda:0')

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
        return out_dict

    def normalize(self, angle):
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def load_from_json(self, input_path):
        fin = open(input_path, 'r')
        j = json.load(fin)
        fin.close()
        self.vertices = j['vertices']
        self.adj_lists = j['edges']

    def add_vertex(self, x, y, theta, cloud=None):
        print('Add new vertex ({}, {}, {}) with idx {}'.format(x, y, theta, len(self.vertices) - 1))
        self.vertices.append((x, y, theta, cloud))
        self.adj_lists.append([])
        if cloud is not None:
            input_data = {'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                     'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda()}
            batch = self._preprocess_input(input_data)
            descriptor = self.model(batch)["final_descriptor"].detach().cpu().numpy()
            self.index.add(descriptor)
        return len(self.vertices) - 1

    def get_k_most_similar(self, cloud, k=1):
        input_data = {'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                     'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda()}
        batch = self._preprocess_input(input_data)
        descriptor = self.model(batch)["final_descriptor"].detach().cpu().numpy()
        _, pred_i = self.index.search(descriptor, k)
        pred_i = pred_i[0]
        pred_tf = []
        for idx in pred_i:
            cand_x, cand_y, cand_theta, cand_cloud = self.vertices[idx]
            cand_cloud_tensor = torch.Tensor(cand_cloud[:, :3]).to(self.device)
            ref_cloud_tensor = torch.Tensor(cloud[:, :3]).to(self.device)
            tf_matrix = self.registration_pipeline.infer(ref_cloud_tensor, cand_cloud_tensor)
            tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
            tf_translation = tf_matrix[:3, 3]
            pred_tf.append(list(tf_rotation) + list(tf_translation))
            print('Tf rotation:', tf_rotation)
            print('Tf translation:', tf_translation)
        print('Pred tf:', np.array(pred_tf))
        return pred_i, np.array(pred_tf)
    
    def add_edge(self, i, j, theta, dst):
        print('Add edge from ({}, {}) to ({}, {})'.format(self.vertices[i][0], self.vertices[i][1], self.vertices[j][0], self.vertices[j][1]))
        self.adj_lists[i].append((j, theta, dst))
        self.adj_lists[j].append((i, self.normalize(theta + np.pi), dst))

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def has_edge(self, u, v):
        for x, _, __ in self.adj_lists[u]:
            if x == v:
                return True
        return False

    def get_path_with_length(self, u, v):
        # Initialize distances and previous nodes dictionaries
        distances = [float('inf')] * len(self.adj_lists)
        prev_nodes = [None] * len(self.adj_lists)
        # Set distance to start node as 0
        distances[u] = 0
        # Create priority queue with initial element (distance to start node, start node)
        heap = [(0, u)]
        # Run Dijkstra's algorithm
        while heap:
            # Pop node with lowest distance from heap
            current_distance, current_node = heapq.heappop(heap)
            if current_node == v:
                path = [current_node]
                cur = current_node
                while cur != u:
                    cur = prev_nodes[cur]
                    path.append(cur)
                path = path[::-1]
                return path, distances[v]
            # If current node has already been visited, skip it
            if current_distance > distances[current_node]:
                continue
            # For each neighbour of current node
            for neighbour, _, weight in self.adj_lists[current_node]:
                # Calculate tentative distance to neighbour through current node
                tentative_distance = current_distance + weight
                # Update distance and previous node if tentative distance is better than current distance
                if tentative_distance < distances[neighbour]:
                    distances[neighbour] = tentative_distance
                    prev_nodes[neighbour] = current_node
                    # Add neighbour to heap with updated distance
                    heapq.heappush(heap, (tentative_distance, neighbour))
        return None, float('inf')
        
    def publish_graph(self):
        graph_msg = MarkerArray()
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
        vertices_marker.color.g = 0
        vertices_marker.color.b = 0
        vertices_marker.color.a = 1
        for x, y, theta, cloud in self.vertices:
            vertices_marker.points.append(Point(x, y, 0.05))
        graph_msg.markers.append(vertices_marker)

        edges_marker = Marker()
        edges_marker.id = 1
        edges_marker.type = Marker.LINE_LIST
        edges_marker.header.frame_id = 'map'
        edges_marker.header.stamp = rospy.Time.now()
        edges_marker.scale.x = 0.05
        edges_marker.color.r = 0
        edges_marker.color.g = 0
        edges_marker.color.b = 1
        edges_marker.color.a = 0.5
        edges_marker.pose.orientation.w = 1
        for u in range(len(self.vertices)):
            for v, theta, dst in self.adj_lists[u]:
                ux, uy, _, __ = self.get_vertex(u)
                vx, vy, _, __ = self.get_vertex(v)
                edges_marker.points.append(Point(ux, uy, 0.05))
                edges_marker.points.append(Point(vx, vy, 0.05))
        graph_msg.markers.append(edges_marker)
        self.pub.publish(graph_msg)

    def save_to_json(self, output_path):
        self.vertices = list(self.vertices)
        for i in range(len(self.vertices)):
            x, y, theta, cloud = self.vertices[i]
            self.vertices[i] = (x, y, theta)
        j = {'vertices': self.vertices, 'edges': self.adj_lists}
        fout = open(output_path, 'w')
        json.dump(j, fout)
        fout.close()