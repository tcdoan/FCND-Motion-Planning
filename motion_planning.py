import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import numpy.linalg as LA

from planning_utils import a_star2, heuristic, create_grid_and_edges, check_hit
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
import pkg_resources
# pkg_resources.require("networkx==2.1")
import networkx as nx

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        colliders = open('colliders.csv', 'r')
        _, lat, _, lon = colliders.readline().replace(',', ' ') .split()
        lat0, lon0 = float(lat), float(lon)            
        self.set_home_position(lon0, lat0, 0.0)
        
        # retrieve current global position
        # convert to current local position using global_to_local()
        local_pos = global_to_local(self.global_position, self.global_home)
        print('local pos {0}'.format(local_pos))

        print('global home {0}, position {1}, local position {2}'.format(
            self.global_home, 
            self.global_position,
            self.local_position))

        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset, edges = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # Convert start position to current position rather than map center
        # grid_start = (local_pos[0] - north_offset, local_pos[1] - east_offset)
        grid_start = (local_pos[0], local_pos[1])
       
        # Set goal as some arbitrary position on the grid
        # grid_goal = (-north_offset + 10, -east_offset + 10)

        # TODO: adapt to set goal as latitude / longitude position and convert

        # Run A* to find a path from start to goal        
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        #       or move to a different search space such as a graph (not done here)
        # path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        G = nx.Graph()

        min_node = edges[0][0]
        min_dist = LA.norm(np.array(grid_start) - np.array(min_node))

        p0 = np.array(grid_start)
        for i in range(len(edges)):
            e = edges[i]
            p1 = e[0]
            p2 = e[1]
            print('edge ', i)
            print('p1 ', p1)
            print('p2 ', p2)
            hit = check_hit(grid, grid_start, p2)
            if hit == False and min_dist > LA.norm(p0 - np.array(p2)):
                min_dist = LA.norm(p0 - np.array(p2))
                min_node = p2

            dist = LA.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        edges.insert(0, (grid_start, min_node))
        G.add_edge(grid_start, min_node, weight=min_dist)

        path = []
        while (len(path) < 1):
            k = np.random.randint(1,len(edges))
            drone_goal = (edges[k][0][0], edges[k][0][1])
            print('Local Start and Goal: ', grid_start, drone_goal)
            path, _ = a_star2(G, heuristic, grid_start, drone_goal)
            
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        for wp in waypoints:
            print('north: ', wp[0], 'east: ', wp[1])

        # Set self.waypoints
        self.waypoints = waypoints

        # send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
