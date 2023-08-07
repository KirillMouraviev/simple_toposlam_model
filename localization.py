import numpy as np

class Localizer():
    def __init__(self, graph, gt_map):
        self.graph = graph
        self.gt_map = gt_map

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

    def localize(self, x, y, theta):
        vertex_ids = []
        rel_poses = []
        dists = []
        for i, v in enumerate(self.graph.vertices):
            dist = np.sqrt((x - v[0]) ** 2 + (y - v[1]) ** 2)
            if dist < 5 and self.gt_map.in_sight(x, y, v[0], v[1]):
                vertex_ids.append(i)
                rel_poses.append(self.get_rel_pose(x, y, theta, v[0], v[1]))
                dists.append(dist)
        ids = list(range(len(dists)))
        ids.sort(key=lambda i: dists[i])
        vertex_ids = [vertex_ids[i] for i in ids]
        rel_poses = [rel_poses[i] for i in ids]
        dists = [dists[i] for i in ids]
        return vertex_ids, rel_poses, dists