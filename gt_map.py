import numpy as np
from skimage.io import imread
from line_of_sight import LineOfSight

class GTMap():
    def __init__(self, gt_map_file):
        gt_map_filename = gt_map_file.split('/')[-1]
        i1, i2, j1, j2 = [int(x) for x in gt_map_filename[12:-4].split('_')]
        self.start_i = i1
        self.start_j = j1
        gt_map = imread(gt_map_file)
        self.gt_map = gt_map
        obstacle_map = (gt_map == 0)
        explored_map = (gt_map != 127)
        grid_map = np.concatenate([explored_map[:, :, np.newaxis], obstacle_map[:, :, np.newaxis]], axis=2)
        self.vis_checker = LineOfSight(grid_map)

    def in_sight(self, x1, y1, x2, y2):
        i1 = int(y1 * 20 + 480 - self.start_i)
        j1 = int(x1 * 20 + 480 - self.start_j)
        i2 = int(y2 * 20 + 480 - self.start_i)
        j2 = int(x2 * 20 + 480 - self.start_j)
        return self.vis_checker.checkLine(i1, j1, i2, j2)