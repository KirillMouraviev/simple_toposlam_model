import json
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from collections import deque

class TopologicalGraph():
    def __init__(self):
        self.vertices = []
        self.adj_lists = []
        self.pub = rospy.Publisher('topological_map', MarkerArray, latch=True, queue_size=100)

    def load_from_json(self, input_path):
        fin = open(input_path, 'r')
        j = json.load(fin)
        fin.close()
        self.vertices = j['vertices']
        self.adj_lists = j['edges']

    def add_vertex(self, x, y, theta, cloud=None):
        print('Add new vertex ({}, {}, {})'.format(x, y, theta))
        self.vertices.append((x, y, theta, cloud))
        self.adj_lists.append([])
        return len(self.vertices) - 1
    
    def add_edge(self, i, j):
        print('Add edge from ({}, {}) to ({}, {})'.format(self.vertices[i][0], self.vertices[i][1], self.vertices[j][0], self.vertices[j][1]))
        self.adj_lists[i].append(j)
        self.adj_lists[j].append(i)

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def has_edge(self, u, v):
        ux, uy, _, __ = self.get_vertex(u)
        vx, vy, _, __ = self.get_vertex(v)
        return v in self.adj_lists[u]

    def get_path_length(self, u, v):
        q = deque()
        dist = [-1] * len(self.adj_lists)
        q.append(u)
        dist[u] = 0
        while len(q) > 0:
            cur = q.popleft()
            if cur == v:
                return dist[cur]
            for to in self.adj_lists[v]:
                if dist[to] == -1:
                    q.append(to)
                    dist[to] = dist[cur] + 1
        return -1

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
            for v in self.adj_lists[u]:
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