import sys
import pkg_resources

# pkg_resources.require("networkx==2.1")
import networkx as nx
import numpy as np
import numpy.linalg as LA
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
import matplotlib.pyplot as plt

class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]

    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)

class Roadmap:
    def __init__(self, data, zmax):
        self._data = data
        extract_polygons()
        self._xmin = np.min(data[:, 0] - data[:, 3])
        self._xmax = np.max(data[:, 0] + data[:, 3])

        self._ymin = np.min(data[:, 1] - data[:, 4])
        self._ymax = np.max(data[:, 1] + data[:, 4])

        self._zmin = 0
        self._zmax = zmax
        self._max_poly_xy = 2 * np.max((data[:, 3], data[:, 4]))
        centers = np.array([p.center for p in self._polygons])
        self._tree = KDTree(centers, metric='euclidean')

    def extract_polygons(self):
        for i in range(self._data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = self._data[i, :]
            obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
            
            # TODO: Compute the height of the polygon
            height = alt + d_alt

            p = Poly(corners, height)
            self._polygons.append(p)

    def sample(self, num_samples):
        """Implemented with a k-d tree for efficiency."""

        xvals = np.random.uniform(self._xmin, self._xmax, num_samples)
        yvals = np.random.uniform(self._ymin, self._ymax, num_samples)
        zvals = np.random.uniform(self._zmin, self._zmax, num_samples)
        samples = list(zip(xvals, yvals, zvals))

        pts = []
        for s in samples:
            in_collision = False
            idxs = list(self._tree.query_radius(np.array([s[0], s[1]]).reshape(1, -1), r=self._max_poly_xy)[0])
            for idx in idxs: 
                p = self._polygons[int(idx)]
                if p.contains(s) and p.height >= s[2]:
                    in_collision = True
            if not in_collision:
                pts.append(s)
                
        return pts

    def can_connect(self, p1, p2):
        ls = LineString([p1, p2])
        for p in self._polygons:
            if p.crosses(ls) and min(p1[2], p2[2]) < p.height:
                return False
        return True

    ## points is list of shapely.geometry.Point(s)
    # k is int param
    def create_graph(self, num_samples, k):
        points = sample(num_samples)
        g = nx.Graph()
        tree = KDTree(points)
        for p in points:
            neighbor_idxs = tree.query([p], k, return_distance=False)[0]
            for i in neighbor_idxs:
                if p !=  points[i] and can_connect(p, points[i]):
                    g.add_edge(p, points[i], weight = LA.norm(np.array(p) - np.array(points[i])))
        return g

    def heuristic(self, position, goal_position):
        return np.linalg.norm(np.array(position) - np.array(goal_position))
