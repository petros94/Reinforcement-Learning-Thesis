import sys
from muscle import *
from math import *
import random
random.seed(0)
from Tiles import fancytiles as tl
from MPIconfig import rank

"""Tile coding settings"""
memsize = 65536*2
numTilings = 16 # numTiles*learning_rate must be < 1, for the RL algorithm to converge

"""Following is LEGACY. If you wish to change settings, do so by altering tl.fancytiles() arguments"""
minInput = [0, 0, 0, 0, 0, 0, 0, 0, 0]
maxInput = [6.3, 1000, 1000, 1000, 1000, 0.5, 0.6, 0.5, 0.6]
tilesPerTiling = [12, 5, 5, 5, 5, 2, 3, 2, 3]
tilingOffset = [0, 0, 0, 0, 0, 0, 0, 0, 0]
tileSize=[0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(tilesPerTiling)):
    tileSize[i] = (maxInput[i]-minInput[i]) / (tilesPerTiling[i] -1) #0.6
    tilingOffset[i] = tileSize[i] / numTilings * (1) #0.6/8


class Robot:
    def __init__(self, P_gain = [500, 500, 150, 400, 200], D_gain = [100, 68, 20, 20, 5]):
        self.robot_controller = Robot_Controller(P_gain, D_gain)
        self.path_planner = Path_Planner(self.robot_controller)
        self.state_memory = []
        self.previous_score = 0

    def tilecodeN(self, arg):
        """LEGACY function"""
        tileIndices = [-1]*numTilings
        x=[0]*len(arg)
        for tiling in range(numTilings):
            for i in range(len(x)):
                x[i] = int(arg[i]/tileSize[i])

            tl=1
            index=0
            for i in range(len(arg)):
                index += x[i]*tl
                tl *= tilesPerTiling[i]
            index += tiling*tl

            for i in range(len(arg)):
                arg[i] += tilingOffset[i]

            tileIndices[tiling] = index

        return tileIndices


    def reset(self, discrete, random_position):
        global sim
        sim.reset()

        """Initial robot config"""
        """Finger angle initial position"""
        id = functions.mj_name2id(model, 3, "f_1_1")
        sim.data.qpos[id] = self.path_planner.max_finger_angles
        id = functions.mj_name2id(model, 3, "f_2_1")
        sim.data.qpos[id] = -self.path_planner.max_finger_angles
        id = functions.mj_name2id(model, 3, "j_3")
        """end-effector Z-axis initial position"""
        sim.data.qpos[3] = 0.0

        """Place object in random orientation"""
        id = 15
        sim.data.qpos[id] = 3.14 if random_position == 0 else np.random.uniform(0, 3.14)
        sim.step()

        """reset robot"""
        x_rel, y_rel, z_rel, distance, angle, finger_angles, touch_points, forces, max_force, spread, statespace = self.path_planner.reset()

        """Get current state depending on which statespace the robot is operating"""
        if statespace == 0:
            float_input_vector = [distance, angle, spread[1]] + forces
            s = tl.fancytiles(numtilings = numTilings, floats = float_input_vector,
                              tilewidths = [0.15, 0.4, 0.2] + [150]*self.robot_controller.n_touch_sensors,
                              memctable = memsize)
            s.append(statespace)
        elif statespace == 1:
            float_input_vector = [x_rel, y_rel, z_rel, angle, spread[1]] + forces
            s = tl.fancytiles(numtilings = numTilings, floats = float_input_vector,
                              tilewidths = [0.1, 0.1, 0.1, 0.6, 0.2] + [150]*self.robot_controller.n_touch_sensors,
                              memctable = memsize)
            s.append(statespace)

        return s

    def step(self, action, discrete):
        """Send action to path planner module and retrieve state variables"""
        x_rel, y_rel, z_rel, distance, current_angle, finger_angles, touch_points, forces, max_force, spread, n_contacts, score, fall, statespace = self.path_planner.step(action)

        """Get current state depending on which statespace the robot is operating"""
        if statespace == 0:
            float_input_vector = [distance, current_angle, spread[1]] + forces
            s = tl.fancytiles(numtilings = numTilings, floats = float_input_vector,
                              tilewidths = [0.15, 0.4, 0.2] + [150]*self.robot_controller.n_touch_sensors,
                              memctable = memsize)
            s.append(statespace)
        elif statespace == 1:
            float_input_vector = [x_rel, y_rel, z_rel, current_angle, spread[1]] + forces
            s = tl.fancytiles(numtilings = numTilings, floats = float_input_vector,
                              tilewidths = [0.1, 0.1, 0.1, 0.6, 0.2] + [150]*self.robot_controller.n_touch_sensors,
                              memctable = memsize)
            s.append(statespace)

        """Identify new states - used for debugging"""
        new_state = 1
        for entry in self.state_memory:
            if any(x in s[0:numTilings] for x in entry):
                new_state = 0
                break

        if new_state:
            self.state_memory.append(s)

        """Diagnostics"""
        global rank
        if rank == 1:
            print("States visited: ", len(self.state_memory))
            print("Input vector: ", float_input_vector)
            # print("State vector:", s)
            # print("Grip score: ", score)
            # print("Force is: ", forces)
            # print("Angle diff: ", angle)
            # print("Laser: ", laser)
            # print("Spread: ", spread)
            sys.stdout.flush()


        """Calculate reward"""
        if action == "EVAL":
            if n_contacts > 1:
                error = self.path_planner.evaluate_grasp()
                reward = 100 if error < 0.07 else -10
                done = 1
                if rank == 1:
                    print("Reward: ", reward, "Error: ", error)
                    sys.stdout.flush()
            else:
                reward = -100
                done = 0
            return s, reward, done


        done = 0

        """Object fell?"""
        if fall == 1:
            reward = -100
            done = 1
            return s, reward, done

        """Reward"""
        if score <= 0:
            reward = -1
        else:
            """Potential based reward"""
            F = 0.9*score - self.previous_score
            reward = F

        self.previous_score = score

        """Diagnostics"""
        if rank == 1:
            print("Reward is: ", reward)
            sys.stdout.flush()
        return s, reward, done
