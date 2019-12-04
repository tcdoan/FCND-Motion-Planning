from enum import Enum
from queue import PriorityQueue
from bresenham import bresenham
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import numpy.linalg as LA

def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """

    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the min, max coordinates we can compute size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])
    
    # Define a list to hold Voronoi points
    points = []
    
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # create a voronoi graph based on location of obstacle centers
    graph = Voronoi(points)
    
    # check each edge from graph.ridge_vertices for collision    
    edges = []           
    for v in graph.ridge_vertices:
        v1 = graph.vertices[v[0]]
        v2 = graph.vertices[v[1]]
        hit = check_hit(grid, v1, v2)
        if not hit:
            edge = (tuple(v1), tuple(v2))
            edges.append(edge)
    
    return grid, int(north_min), int(east_min), edges

def check_hit(grid, v1, v2):
    if np.amin(v1) < 0 or np.amin(v2)< 0 or v1[0] > grid.shape[0] or v1[1] > grid.shape[1] or v2[0] > grid.shape[0] or v2[1] > grid.shape[1]:
        return True
    
    bpoints = list(bresenham(int(v1[0]), int(v1[1]), int(v2[0]), int(v2[1])))
    for p in bpoints:
        if grid[p[0], p[1]] == 1:
            return True

    return False

def a_star(G, h, s, goal):
    queue = PriorityQueue()
    queue.put((0, s))
    visited = set(s)
    
    branch = {}
    found = False    
    while not queue.empty():
        x = queue.get()[1]
        if x == s:
            accum_cost = 0.0
        else:
            accum_cost = branch[x][0]
        
        if x == goal:
            print('Path found!')
            found = True
            break
        else:
            for y in G[x]:
                cost = accum_cost + h(x, y)
                est_cost = cost + h(y, goal)
                
                if y not in visited:
                    visited.add(y)
                    branch[y] = (cost, x)
                    queue.put((est_cost, y))
    
    path = []
    cost = 0
    if found:
        # retrace
        n = goal
        cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != s:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(s)
    else:
        print('Failed to find path!')
    return path[::-1], cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))
