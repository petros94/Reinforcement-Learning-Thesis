import numpy as np
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
from math import *

DATA_Q_values = ["BACK", "FRONT", "OPEN", "CLOSE"]
DATA_actions = []
max_Q = []


class Experiment:
    def __init__(self):
        self.max_episode_number = 10000
        self.episode_number = 0
        self.total_reward = [0]*self.max_episode_number
        self.number_of_steps = [0]*self.max_episode_number
        self.max_Q = [0]*self.max_episode_number
        self.success = [0]*self.max_episode_number
        self.colorr = [0]*self.max_episode_number

    def start(self, episode):
        self.episode_number = episode

    def update(self, reward, maxQ):
        self.total_reward[self.episode_number] += reward
        self.number_of_steps[self.episode_number] += 1
        self.max_Q[self.episode_number] += float(maxQ)

    def stop(self, success):
        if success == "YES":
            self.colorr[self.episode_number] = "green"
            self.success[self.episode_number] = 1
        else:
            self.colorr[self.episode_number] = "red"
            self.success[self.episode_number] = 0

        self.max_Q[self.episode_number] = self.max_Q[self.episode_number] / self.number_of_steps[self.episode_number]

    def results(self):
        time = range(self.episode_number+1)
        plt.scatter(time, self.total_reward[0:self.episode_number+1], color = self.colorr[0:self.episode_number+1])
        plt.pause(0.1)

        plt.figure()
        plt.scatter(time, self.number_of_steps[0:self.episode_number+1], color = self.colorr[0:self.episode_number+1])
        plt.pause(0.1)

    def export(self, path):
        f = open(path, "w+")

        """Headers"""
        f.write("Episode_Number, Total_Reward, Number_of_steps, Success, average_Q, ")
        f.write("\n")

        """Data"""
        for i in range(self.episode_number+1):
            f.write(str(i) + ", ")
            f.write(str(self.total_reward[i]) + ", ")
            f.write(str(self.number_of_steps[i]) + ", ")
            f.write(str(self.success[i]) + ", ")
            f.write(str(self.max_Q[i]) + ", ")
            f.write("\n")

        f.close()


class Failed_Episodes_Data:
    def __init__(self):
        self.total_fails = 0
        self.last_action = []
        self.Q_values = []
        self.cause = []
        self.episode_number = []

    def add(self, episode, last_action, Q_values, cause):
        self.total_fails += 1
        self.episode_number.append(episode)
        self.last_action.append(last_action)
        self.Q_values.append(Q_values)
        self.cause.append(cause)

    def export(self, path):
        f = open(path, "w+")

        """Headers"""
        f.write("Episode_Number, Last_Action, Q_values, Cause,")
        f.write("\n")

        """Data"""
        for i in range(total_fails):
            f.write(str(i) + ", ")
            f.write(str(self.episode_number[i]) + ", ")
            f.write(str(self.last_action[i]) + ", ")
            f.write(str(self.Q_values[i]) + ", ")
            f.write("\n")

        f.close()

def create_info_file(path, agent, state_variables, reward_type, object):
    f = open(path + "info.txt", "w")
    f.write("Agent used: " + agent + "\n")
    f.write("State Variables: " + state_variables + "\n")
    f.write("Reward Used: " + reward_type + "\n")
    f.write("Object: " + object + "\n")



class ObjectEvaluator:
    def __init__(self, sim, model):
        self.id = functions.mj_name2id(model, 1, "red_bull")

        self.time = []

        self.pos_z = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        self.acc_xy = []
        self.acc_theta = []
        self.vel_x = []
        self.vel_y = []
        self.vel_xy = []
        self.vel_z = []
        self.vel_theta = []

        self.time.append(sim.data.time)
        self.pos_z.append(sim.data.get_body_xpos("red_bull")[2])

        acc = np.zeros(6)
        functions.mj_objectAcceleration(model, sim.data, 1, self.id, acc, 0)
        self.acc_x.append(acc[3])
        self.acc_y.append(acc[4])
        self.acc_z.append(acc[5])
        total = pow(acc[3], 2) + pow(acc[4], 2)
        total = sqrt(total)
        self.acc_xy.append(total)
        self.acc_theta.append(acc[2])

        vel = np.zeros(6)
        functions.mj_objectVelocity(model, sim.data, 1, self.id, vel, 0)
        self.vel_x.append(vel[3])
        self.vel_y.append(vel[4])
        self.vel_z.append(vel[5])
        total = pow(vel[3], 2) + pow(vel[4], 2)
        total = sqrt(total)
        self.vel_xy.append(total)
        self.vel_theta.append(vel[2])



    def update(self, sim, model):
        self.time.append(sim.data.time)
        self.pos_z.append(sim.data.get_body_xpos("red_bull")[2])
        acc = np.zeros(6)
        functions.mj_objectAcceleration(model, sim.data, 1, self.id, acc, 0)
        self.acc_x.append(acc[3])
        self.acc_y.append(acc[4])
        self.acc_z.append(acc[5])
        total = pow(acc[3], 2) + pow(acc[4], 2)
        total = sqrt(total)
        self.acc_xy.append(total)
        self.acc_theta.append(acc[2])

        vel = np.zeros(6)
        functions.mj_objectVelocity(model, sim.data, 1, self.id, vel, 0)
        self.vel_x.append(vel[3])
        self.vel_y.append(vel[4])
        self.vel_z.append(vel[5])
        total = pow(vel[3], 2) + pow(vel[4], 2)
        total = sqrt(total)
        self.vel_xy.append(total)
        self.vel_theta.append(vel[2])

    def export(self, path):
        f = open(path, "w")

        """Headers"""
        f.write("No., Time, \
                 Position_Z, \
                 Acceleration_X, Acceleration_Y, Acceleration_Z, Acceleration_XY, Acceleration_Theta, \
                 Velocity_X, Velocity_Y, Velocity_Z, Velocity_XY, Velocity_Theta, ")

        f.write("\n")

        """Data"""
        for i in range(len(self.pos_z)):
            f.write(str(i) + ", ")
            f.write(str(self.time[i]) + ", ")
            f.write(str(self.pos_z[i]) + ", ")
            f.write(str(self.acc_x[i]) + ", ")
            f.write(str(self.acc_y[i]) + ", ")
            f.write(str(self.acc_z[i]) + ", ")
            f.write(str(self.acc_xy[i]) + ", ")
            f.write(str(self.acc_theta[i]) + ", ")
            f.write(str(self.vel_x[i]) + ", ")
            f.write(str(self.vel_y[i]) + ", ")
            f.write(str(self.vel_z[i]) + ", ")
            f.write(str(self.vel_xy[i]) + ", ")
            f.write(str(self.vel_theta[i]) + ", ")
            f.write("\n")

        f.close()


class GraspEvaluator:

    def __init__(self):
        self.finger_geom = [11, 12, 14, 15, 17]
        self.object_geom = [18, 19, 20, 21, 22]
        self.obj_pos = []
        self.obj_theta = 0
        self.obj_mass = 0
        self.obj_inertia = 0
        self.obj_twist = 0
        self.contact_point_list = []
        self.contact_force_3D_list = []
        self.contact_normal_list = []
        self.contact_idx = []
        self.cone_list = []


    def find_contacts(self, sim, model):
        obj_X, obj_Y, obj_Z = sim.data.geom_xpos[18]
        obj_theta = self.get_object_phi_angle(sim)
        for index in range(13):
            contact = sim.data.contact[index]
            if contact.geom1 in self.finger_geom and contact.geom2 in self.object_geom:
                idx = self.finger_geom.index(contact.geom1)
                contact_force_vector = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
                functions.mj_contactForce(model, sim.data, index, contact_force_vector)
                contact_force = contact_force_vector[0:3]
                contact_point = contact.pos
                contact_frame = contact.frame

                self.add_contact_point(contact_point, contact_force, contact_frame)

        self.obj_pos = [obj_X, obj_Y]
        self.obj_theta = obj_theta

        result = np.zeros(6)
        id = functions.mj_name2id(model, 1, "red_bull")
        functions.mj_objectAcceleration(model, sim.data, 1, id, result, 1)

        self.obj_mass = model.body_mass[id]
        self.obj_inertia = model.body_inertia[id][2]
        print(result)
        self.obj_twist = result[5]


    def add_contact_point(self, contact_point, contact_force, contact_frame):
        contact_normal = contact_frame[0:3]
        contact_force_3D = contact_frame[0:3]*contact_force[0] + \
                           contact_frame[3:6]*contact_force[1] + \
                           contact_frame[6:9]*contact_force[2]


        cone = self.calculate_friction_cone(contact_normal, contact_force)

        self.contact_point_list.append(contact_point)
        self.contact_normal_list.append(contact_normal)
        self.contact_force_3D_list.append(contact_force_3D)
        self.cone_list.append(cone)

    def calculate_friction_cone(self, contact_normal, contact_force):
        theta = atan(0.8)
        Fx = contact_force[0]*contact_normal[0]
        Fy = contact_force[0]*contact_normal[1]
        Fz = contact_force[0]*contact_normal[2]

        cone = []
        cone.append([Fx*cos(theta) - Fy*sin(theta), Fx*sin(theta) + Fy*cos(theta)])
        theta = -atan(0.8)
        cone.append([Fx*cos(theta) - Fy*sin(theta), Fx*sin(theta) + Fy*cos(theta)])

        return cone


    def get_object_phi_angle(self, sim):
        current_rot = sim.data.get_geom_xmat("red_bull")
        """get euler angles"""
        if (current_rot[2,0] < 1 and current_rot[2,0] > -1):
            theta = -asin(current_rot[2,0])
            psi = atan2(current_rot[2,1]/cos(theta), current_rot[2,2]/cos(theta))
            phi = atan2(current_rot[1,0]/cos(theta), current_rot[0,0]/cos(theta))
        else:
            phi = 0
        return phi



    def export(self, path):
        f = open(path, "w")

        """Headers"""
        f.write("Contact No. , Contact_Point_X, \
                 Contact_Point_Y, Contact_Point_Z, \
                 Contact_Normal_X, Contact_Normal_Y, \
                 Contact_Normal_Z, Contact_Force_3D_X, \
                 Contact_Force_3D_Y, Contact_Force_3D_Z,\
                 Cone_1_X, Cone_1_Y, \
                 Cone_2_X, Cone_2_Y, \
                 obj_X, obj_Y, \
                 obj_theta, obj_mass, obj_inertia, obj_twist, ")

        f.write("\n")

        """Data"""
        for i in range(len(self.contact_point_list)):
            f.write(str(i) + ", ")
            f.write(str(self.contact_point_list[i][0]) + ", ")
            f.write(str(self.contact_point_list[i][1]) + ", ")
            f.write(str(self.contact_point_list[i][2]) + ", ")

            f.write(str(self.contact_normal_list[i][0]) + ", ")
            f.write(str(self.contact_normal_list[i][1]) + ", ")
            f.write(str(self.contact_normal_list[i][2]) + ", ")


            f.write(str(self.contact_force_3D_list[i][0]) + ", ")
            f.write(str(self.contact_force_3D_list[i][1]) + ", ")
            f.write(str(self.contact_force_3D_list[i][2]) + ", ")

            f.write(str(self.cone_list[i][0][0]) + ", ")
            f.write(str(self.cone_list[i][0][1]) + ", ")
            f.write(str(self.cone_list[i][1][0]) + ", ")
            f.write(str(self.cone_list[i][1][1]) + ", ")

            f.write(str(self.obj_pos[0]) + ", ")
            f.write(str(self.obj_pos[1]) + ", ")
            f.write(str(self.obj_theta) + ", ")

            f.write(str(self.obj_mass) + ", ")
            f.write(str(self.obj_inertia) + ", ")
            f.write(str(self.obj_twist) + ", ")

            f.write("\n")

        f.close()




def write_to_file(path, q_values, action, overwrite):
    global DATA_Q_values
    global DATA_actions
    if overwrite == 0:
        for item in q_values:
            DATA_Q_values.append(item)
        DATA_actions.append(action)
    else:
        f = [open(path[0], "w"), open(path[1], "w")]
        count = 0
        for item in DATA_Q_values:
            f[0].write(str(item) + ", ")
            count += 1
            if count == 4:
                f[0].write("\n")
                count = 0
        for item in DATA_actions:
            f[1].write(item + ", ")

        DATA_Q_values = ["BACK", "FRONT", "OPEN", "CLOSE"]
        DATA_actions = []
        f[0].close()
        f[1].close()

def export_linear_weights(path, weights):
    print("Exporting weights...")
    np.savetxt(path, weights)


def import_linear_weights(path, agent, idx):
    agent.weights[idx] = np.loadtxt(path)



if __name__ == "__main__":

    from math import *
    numTilings = 16
    minInput = [0, 0, 0, 0, 0]
    maxInput = [0.7, 1.2, 1.9, 6.3, 4000]
    tilesPerTiling = [10, 8, 8, 8, 8]
    tilingOffset = [0, 0, 0, 0, 0]
    tileSize=[0 ,0, 0, 0, 0]
    for i in range(len(tilesPerTiling)):
        tileSize[i] = (maxInput[i]-minInput[i]) / (tilesPerTiling[i] -1) #0.6
        tilingOffset[i] = tileSize[i] / numTilings * (1) #0.6/8

    def tilecodeN(arg):
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

    actions = ["BACK", "FRONT", "OPEN_a", "CLOSE_a", "OPEN_b", "CLOSE_b", "LEFT", "RIGHT", "EVAL"]


    size=1
    for i in range(len(tilesPerTiling)):
        size*=tilesPerTiling[i]
    feature_vector = np.array([0]*((numTilings+1)*size), dtype = np.float32)
    SARSA_agent_weights = np.array([[0]*len(feature_vector)]*len(actions), dtype = np.float32)
    SARSA_agent_weights = np.loadtxt("/Users/Petros/Documents/RL_Thesis/Tests/M4_nsarsa_linear_weights.txt")

    distance = np.arange(0.1, 0.7, 0.05)
    relative_angle = np.arange(-0.2, 0.2, 0.1)
    finger_angle_a = np.arange(-0.2, 0.5, 0.05)
    finger_angle_b = np.arange(-0.7, 0, 0.05)
    force = np.arange(0, 4000, 100)

    # size = len(distance)*len(relative_angle)*len(finger_angle_a)*len(finger_angle_b)*len(force)
    # print(size)
    # Q = np.array([[0]*len(actions)]*size, dtype = np.float32)
    # counter = (0)
    #
    # for d in distance:
    #     for fa in finger_angle_a:
    #         for fb in finger_angle_b:
    #             for r in relative_angle:
    #                 for f in force:
    #                     s = tilecodeN([d, fa+0.4, fb+1.57, r+pi, f])
    #                     for action in actions:
    #                         idx = actions.index(action)
    #                         for j in s:
    #                             Q[counter, idx] += SARSA_agent_weights[idx, j]
    #                     counter+=1
    #     print("Progress: ", float(counter)/size*100, "%")
    #
    #
    # np.savetxt("/Users/Petros/Desktop/trash.txt", Q, delimiter = ",", newline="\n")

    finger_angle_a = 0.3
    finger_angle_b = -0.1
    force = 0
    distance = np.arange(0.1, 0.7, 0.01)
    relative_angle = np.arange(-1.5, 1.5, 0.02)
    size = len(relative_angle)*len(distance)
    Q_ = np.array([[0]*(len(actions)+2)]*size, dtype = np.float32)
    counter = 0
    for i in range(len(distance)):
        for j in range(len(relative_angle)):
            s = tilecodeN([distance[i], finger_angle_a+0.4, finger_angle_b+1.57, relative_angle[j]+pi, force])
            for action in actions:
                idx = actions.index(action)
                for k in s:
                    Q_[counter, idx + 2] += SARSA_agent_weights[idx, k]
            Q_[counter, 0] = distance[i]
            Q_[counter, 1] = relative_angle[j]
            counter +=1
        print("Progress: ", float(counter)/size*100, "%")
    np.savetxt("/Users/Petros/Desktop/trash2.txt", Q_, delimiter = ",", newline="\n")
