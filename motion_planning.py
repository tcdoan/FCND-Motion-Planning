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
        print('takeoff transition to ({0}, {1}, {2})'.format(self.local_position[0], self.local_position[1], self.target_position[2]))
        print('Target position ({0}, {1}, {2})'.format(self.target_position[0], self.target_position[1], self.target_position[2]))
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
        TARGET_ALTITUDE = 50
        SAFETY_DISTANCE = 2

        self.target_position[2] = TARGET_ALTITUDE

        colliders = open('colliders.csv', 'r')
        _, lat, _, lon = colliders.readline().replace(',', ' ') .split()
        lat0, lon0 = float(lat), float(lon)            
        self.set_home_position(lon0, lat0, 0)
        
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
        start = (-north_offset + local_pos[0], -east_offset + local_pos[1])

        ## Pick a goal
        # goal_lat = 37.793763
        # goal_long = -122.396823
        path = []
        while len(path) < 2:            
            goal_gps = np.array([-122.402238, 37.797067, TARGET_ALTITUDE])
            # goal_gps = np.array([-122.396823, 37.793763, TARGET_ALTITUDE])
            goal_local = global_to_local(goal_gps, self.global_home)
            print('goal_local', goal_local)

            goal = np.array((-north_offset + goal_local[0], -east_offset + goal_local[1]))
            print('goal', goal)
            noise = np.array((np.random.uniform(-100, 100), np.random.uniform(0, 200)))
            print('noise', noise)

            goal = goal + noise
            print('Goal: ', goal)

            # start2 is the other end inside the Vonoroi graph 
            start2 = np.array(edges[0][0])
            goal2 =   edges[0][1]

            min_start_dist = 1000000.0
            min_goal_dist =  1000000.0
            p0 = np.array(start)
            G = nx.Graph()        
            for i in range(len(edges)):
                p1 = np.array(edges[i][0])
                p2 = np.array(edges[i][1])

                hit = check_hit(grid, p0, p1)
                if hit == False and min_start_dist > LA.norm(p0 - p1):
                    min_start_dist = LA.norm(p0 - p1)
                    start2 = (p1[0], p1[1])

                hit = check_hit(grid, p0, p2)
                if hit == False and min_start_dist > LA.norm(p0 - p2):
                    min_start_dist = LA.norm(p0 - 2)
                    start2 = (p2[0], p2[1])

                hit = check_hit(grid, goal, p1)
                # print('edge {0}, hit {1}, goal {2}, p1 {3}'.format(i, hit, goal, p1))
                if hit == False and min_goal_dist > LA.norm(goal - p1):
                    # print('min_goal_dist: ', min_goal_dist, ', goal2: ', goal2)
                    min_goal_dist = LA.norm(goal - p1)
                    goal2 = (p1[0], p1[1])
                
                hit = check_hit(grid, goal, p2)
                # print('edge {0}, hit {1}, goal {2}, p2 {3}'.format(i, hit, goal, p1))
                if hit == False and min_goal_dist > LA.norm(goal - p2):
                    # print('min_goal_dist: ', min_goal_dist, ', goal2: ', goal2)
                    min_goal_dist = LA.norm(goal - p2)
                    goal2 = (p2[0], p2[1])

            edges.insert(0, (start, start2))        
            for i in range(len(edges)):
                e = edges[i]            
                G.add_edge(e[0], e[1])

            print('Local Start and Goal: ', start, goal2)
            path, _ = a_star2(G, heuristic, start, goal2)
            
        # Convert path to waypoints
        waypoints = [[ int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in path]
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
