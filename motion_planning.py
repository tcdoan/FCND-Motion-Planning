import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import numpy.linalg as LA

from planning_utils import Roadmap
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
        #  GOAL [-122.396823, 37.793763, TARGET_ALTITUDE]
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5
        GOAL_LAT = 37.797067
        GOAL_LON = -122.402238

        self.target_position[2] = TARGET_ALTITUDE

        colliders = open('colliders.csv', 'r')
        _, lat, _, lon = colliders.readline().replace(',', ' ') .split()
        lat0, lon0 = float(lat), float(lon)            
        self.set_home_position(lon0, lat0, 0)
        
        start = global_to_local(self.global_position, self.global_home)
        print('global home {0}, global position {1}, local position {2}, start {3}'.format(self.global_home, self.global_position, self.local_position, start))
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        map = Roadmap(data, 20)
        map.create_graph(300, 7)
        start_neighbors = map.query_close_points(start,5)
        for x in start_neighbors:
            if map.can_connect(start, x):
                map.add_edge(start,x)

        goal_gps = np.array([GOAL_LON, GOAL_LAT, TARGET_ALTITUDE])
        goal = global_to_local(goal_gps, self.global_home)
        goal_neighbors = map.query_close_points(goal,5)
        for y in goal_neighbors:
            if map.can_connect(y, goal):
                map.add_edge(y, goal)

        print('Start finding path from {0} to {1} '.format(start, goal))
        path, _ = map.a_star(start, goal)

        # Convert path to waypoints
        map.map_offsets()
        waypoints = [[ int(p[0]), int(p[1]), TARGET_ALTITUDE, 0] for p in path]
        for wp in waypoints:
            print('north: ', wp[0], 'east: ', wp[1])

        self.waypoints = waypoints
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")
        self.connection.start()
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