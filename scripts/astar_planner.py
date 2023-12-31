import sys
sys.path.append('/home/kirill/TopoSLAM/src/simple_toposlam_model')

from heapq import heappush, heappop
import numpy as np
from line_of_sight import LineOfSight

class Node:
    '''
    Node class represents a search node

    - i, j: coordinates of corresponding grid element
    - g: g-value of the node
    - h: h-value of the node // always 0 for Dijkstra
    - f: f-value of the node // always equal to g-value for Dijkstra
    - parent: pointer to the parent-node 

    '''
    

    def __init__(self, i, j, g = 0, h = 0, f = None, parent = None):
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f        
        self.parent = parent

        
    
    def __eq__(self, other):
        '''
        Estimating where the two search nodes are the same,
        which is needed to detect dublicates in the search tree.
        '''
        return (self.i == other.i) and (self.j == other.j)
    
    def __hash__(self):
        '''
        To implement CLOSED as set of nodes we need Node to be hashable.
        '''
        ij = self.i, self.j
        return hash(ij)


    def __lt__(self, other): 
        '''
        Comparing the keys (i.e. the f-values) of two nodes,
        which is needed to sort/extract the best element from OPEN.
        '''
        return self.f < other.f




class SearchTreePQS: #SearchTree which uses PriorityQueue for OPEN and set for CLOSED
    
    def __init__(self):
        self._open = []   # prioritized queue for the OPEN nodes
        self._closed= set()         # set for the expanded nodes = CLOSED
        self._enc_open_dublicates = 0  # the number of dublicates encountered in OPEN
                                      
    def __len__(self):
        return len(self._open) + len(self._closed)
                
    def open_is_empty(self):
        '''
        open_is_empty should inform whether the OPEN is exhausted or not.
        In the former case the search main loop should be interrupted.
        '''
        return not self._open
    
 
    def add_to_open(self, item):
        '''
        Adding a (previously not expanded) node to the search-tree (i.e. to OPEN).
        It's either a totally new node (the one we never encountered before)
        or it can be a dublicate of the node that currently resides in OPEN.
        It's up to us how to handle dublicates in OPEN. We can either 
        detect dublicates upon adding (i.e. inside this method) or detect them
        lazily, when we are extracting a node for further expansion.
        Not detecting dublicates at all may work in certain setups but let's not
        consider this option.
        '''   
        heappush(self._open, item) #here we ONLY add the node and never check whether the node with the
                                   # same (i,j) resides in PQ already. This leads to occasional duplicates.
                                   # We will check for these dublicates lazily (inside get_best_node)
    

    def get_best_node_from_open(self):
        '''
        Extracting the best node (i.e. the one with the minimal key) from OPEN.
        This node will be expanded further on in the main loop of the search.
        ''' 
        #print("retrieving the best node:")
        best_node = heappop(self._open)
        while self.was_expanded(best_node): #this line checks whether we have retrieved a duplicate. If yes - move on.
            #print('Node ',best_node.i, best_node.j,' is a dublicate.')
            
            if not self._open: #this happens when only dublicates are left in PQ (=task not solvable)
                return None
            
            best_node = heappop(self._open)
            self._enc_open_dublicates +=1
        #print("The best node is: ",best_node.i, best_node.j,". It's g=", best_node.g)
        return best_node

    def add_to_closed(self, item):
        self._closed.add(item)

    def was_expanded(self, item):
        return item in self._closed

    @property
    def opened(self):
        return self._open
    
    @property
    def expanded(self):
        return self._closed

    @property
    def number_of_open_dublicates(self):
        return self._enc_open_dublicates




def computeHFromCellToCell(i1, j1, i2, j2):
    return np.sqrt((i2 - i1) ** 2 + (j2 - j1) ** 2)


class AStarPlanner:
    def __init__(self, grid, allow_diagonal=True):
        self.hweight = 1
        self.grid = grid
        obstacle_map = (self.grid == 0)
        explored_map = (self.grid != 127)
        grid_map = np.concatenate([explored_map[:, :, np.newaxis], obstacle_map[:, :, np.newaxis]], axis=2)
        self.vis_checker = LineOfSight(grid_map)
        self.allow_diagonal = allow_diagonal

    def reset(self):
        pass


    def getClosestNode(self, x, y):
        i, j = self.mapper.world_to_map(x, y)
        return Node(i, j)


    def findSuccessors(self, node):
        successors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if not self.allow_diagonal and di * dj != 0:
                    continue
                if di == 0 and dj == 0:
                    continue
                if node.i + di < 0 or node.i + di >= self.grid.shape[0]:
                    continue
                if node.j + dj < 0 or node.j + dj >= self.grid.shape[1]:
                    continue
                if not self.vis_checker.checkTraversability(node.i + di, node.j + dj):
                    continue
                successors.append((node.i + di, node.j + dj))
        return successors


    def restore_path(self, current):
        path = []
        while current is not None:
            path.append((current.i, current.j))
            current = current.parent
        path.reverse()
        return path


    def smooth_path(self, path):
        path_smoothed = [path[0]]
        for k in range(2, len(path)):
            i, j = path[k]
            if not self.vis_checker.checkLine(path_smoothed[-1][0], path_smoothed[-1][1], i, j):
                path_smoothed.append(path[k - 1])
        path_smoothed.append(path[-1])
        return path_smoothed


    def computeCost(self, i1, j1, i2, j2):
        dst = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)
        if self.grid[i2, j2] == 127:
            return dst * self.unknown_cost
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                if i2 + di < 0 or i2 + di >= self.grid.shape[0]:
                    continue
                if j2 + dj < 0 or j2 + dj >= self.grid.shape[1]:
                    continue
        return dst


    def create_path(self, start, goal, unknown_is_obstacle=True, goal_reacher=False):
        self.vis_checker.unknownIsObstacle = unknown_is_obstacle
        start_i, start_j = start
        goal_i, goal_j = goal
        queue = SearchTreePQS()

        start_node = Node(start_i, start_j)
        goal_node = Node(goal_i, goal_j)
        if not self.vis_checker.checkTraversability(start_i, start_j) or not self.vis_checker.checkTraversability(goal_i, goal_j):
            return None
        start_node.h = computeHFromCellToCell(start_node.i, start_node.j, goal_node.i, goal_node.j)
        start_node.f = self.hweight * start_node.h
        start_node.parent = None
        queue.add_to_open(start_node)
        path_found = False

        while not queue.open_is_empty():
            current = queue.get_best_node_from_open()
            if current is None:
                break
            queue.add_to_closed(current)

            if current.h < 2:
                path_found = True
                break

            for i, j in self.findSuccessors(current):
                new_node = Node(i, j)
                if not queue.was_expanded(new_node):
                    new_node.g = current.g + self.computeCost(current.i, current.j, i, j)
                    new_node.h = computeHFromCellToCell(i, j, goal_node.i, goal_node.j)
                    new_node.f = new_node.g + self.hweight * new_node.h
                    new_node.parent = current
                    queue.add_to_open(new_node)

        if path_found:
            path = self.restore_path(current)
            path = self.smooth_path(path)
            return path

        return None