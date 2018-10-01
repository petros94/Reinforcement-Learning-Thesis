import numpy as np
import scipy as sc
import scipy.linalg
from math import *
from MPIconfig import rank
import sys
sys.path.append("/Users/Petros/Documents/RL_Thesis/Python/mujoco-py")

from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
from pyhull.convex_hull import ConvexHull
from pulp import *


fullpath = "/Users/Petros/Desktop/RL_Grasping_Thesis/hello_world5.xml"
model = load_model_from_path(fullpath)
#
sim = MjSim(model)
viewer = MjViewer(sim)


class Robot_Controller:

    class Contact_point:
        """
        This class defines a contact point. An object is initialized using
        data from mujoco
        """

        def __init__(self, force, frame, geom, index, coordinates, obj_geom):
            self.force = force          #Force Magnitude
            self.frame = frame[0:3]     #Contact normal vector
            self.geom = geom            #Robot's geom involved
            self.obj_geom = obj_geom    #Obj's geom involved
            self.index = index          #index of finger (finger_geom list)
            self.coords = coordinates   #Coordinates of contact
            self.r = np.array([self.coords[0] - sim.data.subtree_com[13][0],
                               self.coords[1] - sim.data.subtree_com[13][1],
                               self.coords[2] - sim.data.subtree_com[13][2]], dtype = np.float32)       #Distance from center of body mass

            self.mu = 0.4               #Contact friction
            self.wrench = [np.array([0,0,0], dtype = np.float32), np.array([0,0,0], dtype = np.float32), np.array([0,0,0], dtype = np.float32)] #Stores contact wrench
            self.calculate_wrench()     #Calculates wrench

        def calculate_wrench(self):
            theta = atan(self.mu)
            Fx = self.force*self.frame[0]
            Fy = self.force*self.frame[1]
            Fz = self.force*self.frame[2]

            self.wrench[0][0] = Fx*cos(theta) - Fy*sin(theta)
            self.wrench[0][1] = Fx*sin(theta) + Fy*cos(theta)
            self.wrench[0][2] = -self.wrench[0][0]*self.r[1] + self.wrench[0][1]*self.r[0]


            theta = -atan(self.mu)
            self.wrench[1][0] = Fx*cos(theta) - Fy*sin(theta)
            self.wrench[1][1] = Fx*sin(theta) + Fy*cos(theta)
            self.wrench[1][2] = -self.wrench[1][0]*self.r[1] + self.wrench[1][1]*self.r[0]


            self.wrench[2][0] = Fx
            self.wrench[2][1] = Fy
            self.wrench[2][2] = -self.wrench[2][0]*self.r[1] + self.wrench[2][1]*self.r[0]


    def __init__(self, P_gain, D_gain):
        self.pose = []
        self.rot = []
        self.joints = ["j_0", "j_1", "j_2", "j_3", "j_4", "f_1_1",
             "f_2_1","f_1_2", "f_2_2"]                  #List of joints
        self.n_joints = len(self.joints)                #Num. joints
        self.jacobian = np.zeros([4, self.n_joints-5])  #Robot Jacobian
        self.dof = len(self.joints)+6                   #Dof (Robot + object)
        self.qvel = np.zeros(self.n_joints)             #Joint velocities

        self.desired_values = [0]*(5 + 4)               #Desired values for joints: 5 for robot arm and 4 for finger joints
        self.P_gain = P_gain                            #Controller P Gain
        self.D_gain = D_gain                            #Controller D Gain
        self.P_gain_matrix = np.array([P_gain[0], 0, 0, 0,
                                       0, P_gain[1], 0, 0,
                                       0, 0, P_gain[2], 0,
                                       0, 0, 0, P_gain[3]])

        self.D_gain_matrix = np.array([D_gain[0], 0, 0, 0,
                                       0, D_gain[1], 0, 0,
                                       0, 0, D_gain[2], 0,
                                       0, 0, 0, D_gain[3]])
        self.P_gain_matrix.shape = [4,4]
        self.D_gain_matrix.shape = [4,4]

        """Contact detection"""
        self.finger_geom = [11, 12, 14, 15, 17]         #Geom number of fingers
        self.object_geom = [18, 19, 20, 21, 22]         #Geom number of object
        self.obstacle_geom = [23]                       #Geom number of obstacle if used
        self.contact_points = [0]*len(self.finger_geom) #list of contact points
        self.last_contact = [0]*len(self.finger_geom)   #list of past contact points - used to identify different contacts
        self.n_touch_sensors = 10                       #Number of touch sensors - five in each finger
        self.n_laser_sensors = 3                        #Laser sensors - not used

        """grasp eval"""
        self.rise_force = 0                             #Flag used for raising the arm
        self.convex_option = 1                          #Flag used to specify grasp score method - 0 = inner product, 1 = convex hull, 2 = no score


        """Object stabilization"""
        self.acc_stabilization = 1                      #Used to determine if the object is stable
        self.vel_stabilization = 1                      #Used to determine if the object is stable



    def get_jacob0(self):
        """
        Get the jacobian at the current position
        """
        global sim
        jacp = sim.data.get_geom_jacp("end_effector_base")
        jacr = sim.data.get_geom_jacr("end_effector_base")
        for i in range(self.n_joints-5):
            self.jacobian[0,i] = jacp[i]
            self.jacobian[1,i] = jacp[i + self.dof]
            self.jacobian[2,i] = jacp[i + 2*self.dof]
            self.jacobian[3,i] = jacr[i + 2*self.dof]
        return self.jacobian

    def get_pose(self):
        """
        Get x-y-z coordinates of end effector, expressed in world frame.
        """
        global sim
        self.pose = sim.data.get_geom_xpos("end_effector_base")
        return self.pose

    def get_rot(self):
        """
        Get rotation matrix of end effector
        """
        global sim
        self.rot = sim.data.get_geom_xmat("end_effector_base")
        return self.rot

    def get_joint_velocities(self):
        global sim
        for i in range(self.n_joints):
            self.qvel[i] = sim.data.get_joint_qvel(self.joints[i])
        return self.qvel

    def get_joint_position(self, index):
        global sim
        return sim.data.get_joint_qpos(self.joints[index])

    """For the following functions, the formatting used is: get_finger_angle_FINGERNUMBER_JOINTNUMBER()"""

    def get_finger_angle_1_1(self):
        global sim
        return sim.data.get_joint_qpos("f_1_1")

    def get_finger_angle_1_2(self):
        global sim
        return sim.data.get_joint_qpos("f_1_2")

    def get_finger_angle_2_1(self):
        global sim
        return sim.data.get_joint_qpos("f_2_1")

    def get_finger_angle_2_2(self):
        global sim
        return sim.data.get_joint_qpos("f_2_2")

    def get_finger_angles(self):
        angles = [self.get_finger_angle_1_1(), self.get_finger_angle_1_2(),
                  self.get_finger_angle_2_1(), self.get_finger_angle_2_2()]
        return angles

    def get_euler_angle(self):
        """
        Return euler angles of end effector Z-Y'-X'. Only the phi angle is actually used, since the problem is planar.
        """
        phi = 0
        current_rot = self.get_rot()
        """get euler angles"""
        if (current_rot[2,0] < 1 and current_rot[2,0] > -1):
            theta = -asin(current_rot[2,0])
            psi = atan2(current_rot[2,1]/cos(theta), current_rot[2,2]/cos(theta))
            phi = atan2(current_rot[1,0]/cos(theta), current_rot[0,0]/cos(theta))
        else:
            phi = 0
            if (current_rot[2,0] <= -1):
                theta = pi/2
                psi = phi + atan2(current_rot[0,1], current_rot[0,2])
            else:
                theta = -pi/2
                psi = phi - atan2(-current_rot[0,1], -current_rot[0,2])
        return psi, theta, phi

    def get_object_phi_angle(self):
        current_rot = sim.data.get_geom_xmat("red_bull")
        """get euler angles"""
        if (current_rot[2,0] < 1 and current_rot[2,0] > -1):
            theta = -asin(current_rot[2,0])
            psi = atan2(current_rot[2,1]/cos(theta), current_rot[2,2]/cos(theta))
            phi = atan2(current_rot[1,0]/cos(theta), current_rot[0,0]/cos(theta))
        else:
            phi = 0
        return phi


    def controller_update(self):
        """
        This function updates the controller's output, based on the error between desired and actual joint values.
        The method used to transit from the end-effector space to the joint space is called Damped Least Squares
        """
        outputs = np.zeros(self.n_joints)

        """desired_values = pose_values x-y-z-phi (4) , finger_angles (4)"""
        J = self.get_jacob0()
        lamda = 0.001
        Q = np.matmul(np.transpose(J), J) + lamda*lamda*np.array([[1,0,0,0],
                                                                  [0,1,0,0],
                                                                  [0,0,1,0],
                                                                  [0,0,0,1]])
        Q = scipy.linalg.inv(Q)
        Q = np.matmul(Q, np.transpose(J))

        current_pose = self.get_pose()
        qvel = self.get_joint_velocities()

        """get euler angles"""
        psi, theta, phi = self.get_euler_angle()

        v = np.matmul(Q, self.desired_values[0:4] - np.concatenate([current_pose[0:3], [phi]]))
        outputs[0:(self.n_joints-5)] = np.matmul(self.P_gain_matrix, v) - np.matmul(self.D_gain_matrix, qvel[0:4])
        outputs[4] = self.P_gain[4]*(self.desired_values[4] - self.get_joint_position(4)) - self.D_gain[4]*qvel[4]
        outputs[5] = 200*(self.desired_values[5] - self.get_joint_position(5)) -15*qvel[5]
        outputs[6] = 200*(-self.desired_values[6] - self.get_joint_position(6)) -15*qvel[6]
        outputs[7] = 200*(self.desired_values[7] - self.get_joint_position(7)) -15*qvel[7]
        outputs[8] = 200*(-self.desired_values[8] - self.get_joint_position(8)) -15*qvel[8]

        """if evaluation mode"""
        if self.rise_force > 0:
            outputs[3] = 10*self.rise_force

        """update simulator"""
        global sim
        for i in range(9):
            sim.data.ctrl[i] = outputs[i]



    def detect_contact(self):
        """
        This function is responsible for detecting the contact forces between the robot's fingers and the object, in every simulation loop.
        It also calculates the score of the grasp based on the prefered method (inner product/ convex hull) and checks for finger collisions
        """
        global sim
        counter = 0
        contact_force = [0]*len(self.finger_geom)
        contact_list = []
        contact_frame = 0
        finger_collision = 0
        for index in range(13):
            contact = sim.data.contact[index]

            """Check if this contact involves any of the robot's fingers and the object"""
            if contact.geom1 in self.finger_geom and contact.geom2 in self.object_geom:

                idx = self.finger_geom.index(contact.geom1)

                """
                Check if this contact still exists. In mujoco, a contact remains in the contacts table, until it gets replaced. This means that we have to check if the contact is still relevant,
                by checking the contact.dist parameter. When a contact is present across multiple simulation steps, this parameter oscillates a bit, due to simulation imperfections. So if a contact is inactive,
                the parameter will have the same value between two successive steps. We use this property to identify active contacts
                """

                if contact.dist != self.last_contact[idx]:
                    self.contact_points[idx] = contact.dist
                    counter +=1
                    contact_force_vector = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
                    functions.mj_contactForce(model, sim.data, index, contact_force_vector)
                    contact_force[idx] = contact_force_vector[0]
                    contact_list.append(self.Contact_point(force = contact_force_vector[0], frame = contact.frame, geom = contact.geom1, index = idx, coordinates = contact.pos, obj_geom = contact.geom2))
                    sim.data.contact[index].geom1 = 99

            """Check for finger collision """
            if contact.geom1 in self.finger_geom and contact.geom2 in self.finger_geom:
                finger_collision = 1
                break

        for i in range(len(self.finger_geom)):
            self.last_contact[i] = self.contact_points[i]

        score = 0
        hull = 0


        """Calculate Inner Product/ Convex Hull score"""
        if self.convex_option == 0:
            for i in range(len(contact_list) -1):
                for j in range(i+1, len(contact_list)):
                    a = contact_list[i].frame*contact_list[i].force
                    b = contact_list[j].frame*contact_list[j].force
                    prod = np.inner(a,b)
                    score += -prod

            score =  1 - np.exp(-2*score/160000)
            score =  max(score, 0)

        elif self.convex_option == 1:
            pts = []
            for contact in contact_list:
                pts.append([contact.wrench[0][0], contact.wrench[0][1], contact.wrench[0][2]])
                pts.append([contact.wrench[1][0], contact.wrench[1][1], contact.wrench[1][2]])
                pts.append([contact.wrench[2][0], contact.wrench[2][1], contact.wrench[2][2]])

            if len(pts)>5 and pts:
                try:
                    hull = ConvexHull(pts)
                except AttributeError:
                    pass

        return counter, contact_force, score, finger_collision, hull

    def get_sensor_data(self):
        global sim
        touch = []
        forces = []
        max_force = 0

        for i in range(self.n_touch_sensors):
            output = 1 if sim.data.sensordata[i] > 0 else 0
            if max_force < sim.data.sensordata[i]:
                max_force = sim.data.sensordata[i]
            touch.append(output)
            forces.append(sim.data.sensordata[i])

        return touch, forces, max_force


    def get_point_of_contact(self, contact):
        point = np.array(contact.coords)
        point.shape = 3,1
        geom = contact.geom
        rotation = np.array(sim.data.geom_xmat[geom])
        rotation.shape = 3,3
        origin = np.array(sim.data.geom_xpos[geom])
        origin.shape = 3,1
        point_ = np.matmul(np.transpose(rotation), point - origin)
        return point_[2][0]


    def orthogonal_distance(self, hull_polygon, point):
        """Returns distance between point and plane"""
        min_d = 1000
        for normal_vector in hull_polygon:
            d = normal_vector[0]*point[0] + normal_vector[1]*point[1] + normal_vector[2]*point[2] + normal_vector[3]
            if abs(d) < min_d:
                min_d = abs(d)
        return min_d

    def contact_score(self):
        """LEGACY - not used"""
        global sim
        contacts = sim.data.contact
        contact_list = []
        for c in contacts:
            if c.geom2 in self.object_geom and c.geom1 in self.finger_geom:
                c.geom1 = 99
                contact_force_vector = np.zeros(6, dtype=np.float64)
                functions.mj_contactForce(model, sim.data, contacts.index(c), contact_force_vector)
                contact_list.append(self.Contact_point(force = contact_force_vector[0], frame = c.frame, geom = c.geom1))

        score = 0
        for i in range(len(contact_list)-1):
            for j in range(i+1, len(contact_list)):
                a = contact_list[i].frame*contact_list[i].force
                b = contact_list[j].frame*contact_list[j].force
                prod = np.inner(a,b)
                score += -prod

        score = 1 - np.exp(-score/160000)
        score = max(score, 0)
        return score


    def object_fall(self):
        """Detect if object has become unstable/out of bounds, by checking its vertical orientation/ distance from center"""
        global sim
        global rank
        R = sim.data.get_body_xmat("red_bull")
        pos_1 = sim.data.get_body_xpos("red_bull")
        pos_2 = sim.data.get_body_xpos("end_effector")
        if pos_1[0] > 4 or pos_1[0] < 0.5 or pos_1[1] > 1.5 or pos_1[1] < -1.5 :
            if rank == 1:
                print("Object out of bounds")
                sys.stdout.flush()
            return 1
        # if self.distance_from_object() > 0.5:
        #     print("Object out of bounds")
        #     return 1
        # else:
        #     return 0
        if R[2,2] < 0.8:
            if rank == 1:
                print("Object fell")
                sys.stdout.flush()
            return 1
        else:
            return 0

    def get_object_pose(self):
        global sim
        return sim.data.get_body_xpos("red_bull")

    def distance_from_object(self):
        global sim
        robot = np.array(sim.data.get_geom_xpos("end_effector_base"))
        red_bull = np.array(sim.data.get_body_xpos("red_bull"))
        x = pow(robot[0]-red_bull[0], 2)
        y = pow(robot[1]-red_bull[1], 2)
        return sqrt(x+y)

    def get_relative_angle(self):
        """Get object and robot coordinates"""
        x_obj, y_obj = self.get_object_pose()[0:2]
        x_robot, y_robot = self.get_pose()[0:2]

        """Calculate relative angle"""
        relative_angle = atan2(y_robot - y_obj, x_robot - x_obj)

        return relative_angle

    def get_relative_angle2(self):
        """Get object and robot coordinates"""
        x_obj, y_obj = self.get_object_pose()[0:2]
        x_robot, y_robot = self.get_pose()[0:2]

        """Calculate relative angle"""
        relative_angle = atan2(y_obj-y_robot, x_obj-x_robot)

        return relative_angle

    def force_closure_test(self, vertices):
        """Test whether the contacts form a force closure"""
        size = len(vertices)
        origin = vertices[0]
        dim = len(origin)
        new_vectors = np.zeros([size-1, dim], dtype = np.float32)
        for i in range(0, size-1):
            for j in range(dim):
                new_vectors[i][j] = vertices[i+1][j] - origin[j]

        """form an lp problem"""
        prob = LpProblem("convex_test", LpMinimize)
        lp_variables = []
        for i in range(size-1):
            name = "k" + str(i)
            lp_variables.append(LpVariable(name, 0, 1))

        prob += lpSum(lp_variables)
        for i in range(dim):
            condition = 0
            for j in range(size-1):
                condition += new_vectors[j][i]*lp_variables[j]
            condition = condition == -origin[i]
            prob += condition

        prob += lpSum(lp_variables) <= 1
        solution = prob.solve()
        return solution

    def object_stabilized(self):
        """Check if object is stabilized. Used in order to determine when the next action will take place"""
        result = np.zeros(6)
        id = functions.mj_name2id(model, 1, "red_bull")
        functions.mj_objectAcceleration(model, sim.data, 1, id, result, 0)
        self.acc_stabilization = 0.99*self.acc_stabilization + 0.01*result
        for entry in self.acc_stabilization[0:5]:
            if abs(entry) > 0.6:
                return 1
        functions.mj_objectVelocity(model, sim.data, 1, id, result, 0)
        for entry in result:
            if abs(entry) > 0.1:
                return 1
        self.acc_stabilization = 1
        return 0

    def get_spread(self):
        """Get finger spread"""
        point_1 = np.array(sim.data.site_xpos[4]) #4 5
        point_2 = sim.data.site_xpos[9]          #9 11
        idx = functions.mj_name2id(model, 1, "end_effector")
        rotation = sim.data.body_xmat[idx]

        spread = [0,0]
        spread[0] = ((point_1[0]-point_2[0])*rotation[0]+(point_1[1]-point_2[1])*rotation[3]+(point_1[2]-point_2[2])*rotation[6])
        spread[1] = abs((point_1[0]-point_2[0])*rotation[1]+(point_1[1]-point_2[1])*rotation[4]+(point_1[2]-point_2[2])*rotation[7])
        return spread






class Path_Planner:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.statespace = 0
        self.max_finger_angles = 0.5 #defaut is 0.5

    def reset(self):

        """reset PID controllers and position"""
        current_position = self.robot_controller.get_pose()
        current_angle = self.robot_controller.get_euler_angle()
        contact, force, score, finger_collision, hull = self.robot_controller.detect_contact()
        self.robot_controller.desired_values[0] = current_position[0]
        self.robot_controller.desired_values[1] = current_position[1]
        self.robot_controller.desired_values[2] = current_position[2]
        self.robot_controller.desired_values[4] = current_angle[0]
        self.robot_controller.desired_values[3] = current_angle[1]
        self.robot_controller.desired_values[5] = self.robot_controller.get_finger_angle_1_1()
        self.robot_controller.desired_values[6] = -self.robot_controller.get_finger_angle_2_1()
        self.robot_controller.desired_values[7] = self.robot_controller.get_finger_angle_1_2()
        self.robot_controller.desired_values[8] = -self.robot_controller.get_finger_angle_2_2()

        """Reset rise force"""
        self.robot_controller.rise_force = 0

        """State variables"""
        self.robot_controller.distance = self.robot_controller.distance_from_object()
        self.robot_controller.relative_angle = self.robot_controller.get_relative_angle()
        distance = self.robot_controller.distance
        touch_points, forces, max_force = self.robot_controller.get_sensor_data()
        finger_angles = self.robot_controller.get_finger_angles()
        angle = self.robot_controller.get_euler_angle()[2] - self.robot_controller.get_object_phi_angle() +2*pi
        spread = self.robot_controller.get_spread()
        x_obj, y_obj, z_obj = self.robot_controller.get_object_pose()[0:3]
        x_rel = current_position[0] - x_obj
        y_rel = current_position[1] - y_obj
        z_rel = current_position[2] - z_obj

        self.statespace = 0

        return x_rel, y_rel, z_rel, distance, angle, finger_angles, touch_points, forces, max_force, spread, self.statespace


    def step(self, action):
        global sim
        global viewer

        """Get end effector position"""
        current_position = self.robot_controller.get_pose()

        """Get end_effector engle"""
        current_angle = self.robot_controller.get_euler_angle()[2]

        """Get object and robot coordinates"""
        x_obj, y_obj, z_obj  = self.robot_controller.get_object_pose()[0:3]
        x_robot, y_robot, z_robot = current_position[0:3]

        """Get distance from object"""
        distance = self.robot_controller.distance_from_object()

        """Calculate relative angle"""
        relative_angle = atan2(y_robot - y_obj, x_robot - x_obj)


        """Wait for PID controller to stablilize output"""
        t=0
        contact = 0
        fall = 0
        score = 0
        hull = 0
        finger_collision = 0
        temp_forces = []
        touch_points = []
        max_force = 0
        max_t = 200
        while (t<max_t or self.robot_controller.object_stabilized()):
            self.action_decode(action, min(t,max_t), max_t, distance, relative_angle, current_angle, x_obj, y_obj, z_obj, x_robot, y_robot, z_robot)
            self.robot_controller.controller_update()
            if t > max_t-3:
                contact, force, score, finger_collision, hull = self.robot_controller.detect_contact()
                touch_points, forces,  max_force = self.robot_controller.get_sensor_data()
                temp_forces.append(forces)
            sim.step()
            if self.robot_controller.object_fall() or finger_collision:
                fall = 1
                break
            viewer.render()


            t+=1
            if t>4000:
                break


        """Filter touch sensor noise"""
        forces = []
        for entry in list(zip(*temp_forces)):
            forces.append(min(np.mean(entry), 750))


        """Detect force closure"""
        if self.robot_controller.convex_option == 1 and hull != 0:
            if self.robot_controller.force_closure_test(hull.vertices):
                """If contact has force closure, measure reward based on convex hull size"""
                score = 1 - np.exp(-2/15*self.robot_controller.orthogonal_distance(hull.polygon, [0,0,0]))
            else:
                score = 0

        if self.robot_controller.convex_option == 2:
            score = -1


        """Get current state variables"""
        distance = self.robot_controller.distance_from_object()
        current_angle = self.robot_controller.get_euler_angle()[2] - self.robot_controller.get_object_phi_angle() +2*pi
        finger_angles = [self.robot_controller.get_finger_angle_1_1(), self.robot_controller.get_finger_angle_1_2(),
                         self.robot_controller.get_finger_angle_2_1(), self.robot_controller.get_finger_angle_2_2()]
        spread = self.robot_controller.get_spread()
        n_contacts = contact

        x_obj, y_obj, z_obj  = self.robot_controller.get_object_pose()[0:3]
        x_robot, y_robot, z_robot = current_position[0:3]
        x_rel = x_robot - x_obj
        y_rel = y_robot - y_obj
        z_rel = z_robot - z_obj

        return x_rel, y_rel, z_rel, distance, current_angle, finger_angles, touch_points, forces, max_force, spread, n_contacts, score, fall, self.statespace


    def evaluate_grasp(self):
        """Used when "EVAL" action is taken"""
        global sim
        global viewer
        global rank

        if rank == 1:
            print("Evaluating grasp...")
            sys.stdout.flush()


        """calculate initial differcnce between contact points"""
        z_obj = sim.data.get_body_xpos("red_bull")[2]
        z_robot = sim.data.get_body_xpos("end_effector")[2]
        bias = z_obj - z_robot

        """Rise arm to maximum height"""
        t = 0
        error=0
        max_t = 200
        initial_z = self.robot_controller.desired_values[2]
        initial_obs = [sim.data.get_body_xpos("red_bull")[0], sim.data.get_body_xpos("red_bull")[1]]
        while t<2000:
            self.robot_controller.desired_values[2] = (0.8 - initial_z)*min(t, max_t)/max_t + initial_z
            self.robot_controller.controller_update()

            """Calculate object velocity"""
            u_obs = (sim.data.get_body_xpos("red_bull")[2] - z_obj)/0.001

            """Calculate end-effector velocity"""
            u_robot = (sim.data.get_body_xpos("end_effector")[2] - z_robot)/0.001

            diff = abs(u_obs - u_robot)
            error += diff*0.001

            z_obj = sim.data.get_body_xpos("red_bull")[2]
            z_robot = sim.data.get_body_xpos("end_effector")[2]

            sim.step()
            viewer.render()
            t+=1


        final_obs = [sim.data.get_body_xpos("red_bull")[0], sim.data.get_body_xpos("red_bull")[1]]
        x_error = final_obs[0] - initial_obs[0]
        y_error = final_obs[1] - initial_obs[1]
        error = abs((z_obj - z_robot) - bias) + abs(x_error) + abs(y_error)

        return abs(error)


    def change_statespace(self):
        """Used to switch between statespaces"""
        global sim
        global viewer
        global rank

        x_obj, y_obj, z_obj  = self.robot_controller.get_object_pose()[0:3]

        self.robot_controller.desired_values[2] = 0.7
        self.robot_controller.desired_values[0] = x_obj
        self.robot_controller.desired_values[1] = y_obj
        self.robot_controller.desired_values[4] = pi/2 - 0.01
        self.statespace = 1


    def action_decode(self, action, t, max_step, distance, relative_angle, current_angle, x_obj, y_obj, z_obj, x_robot, y_robot, z_robot):
        """Turns actions to desired end-effector position and orientation"""
        if action == "S1_FRONT":
            delta_r = -0.15
            self.robot_controller.distance = distance + delta_r
            self.robot_controller.desired_values[0] = x_obj + self.robot_controller.distance*cos(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[1] = y_obj + self.robot_controller.distance*sin(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[3] = self.robot_controller.relative_angle +pi if self.robot_controller.relative_angle < 0 else self.robot_controller.relative_angle - pi

        elif action == "S1_BACK":
            if distance < 0.85:
                delta_r = 0.15
            else:
                delta_r = 0
            self.robot_controller.distance = distance + delta_r
            self.robot_controller.desired_values[0] = x_obj + self.robot_controller.distance*cos(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[1] = y_obj + self.robot_controller.distance*sin(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[3] = self.robot_controller.relative_angle +pi if self.robot_controller.relative_angle < 0 else self.robot_controller.relative_angle - pi

        elif action == "S1_LEFT":
            delta_phi = 0.4*t/max_step
            self.robot_controller.relative_angle = relative_angle+ delta_phi
            if self.robot_controller.relative_angle > pi:
                self.robot_controller.relative_angle -= 2*pi
            elif self.robot_controller.relative_angle < -pi:
                self.robot_controller.relative_angle += 2*pi

            self.robot_controller.desired_values[0] = x_obj + self.robot_controller.distance*cos(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[1] = y_obj + self.robot_controller.distance*sin(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[3] = self.robot_controller.relative_angle +pi if self.robot_controller.relative_angle < 0 else self.robot_controller.relative_angle - pi

        elif action == "S1_RIGHT":
            delta_phi = -0.4*t/max_step
            self.robot_controller.relative_angle = relative_angle+ delta_phi
            if self.robot_controller.relative_angle < -pi:
                self.robot_controller.relative_angle += 2*pi
            elif self.robot_controller.relative_angle > pi:
                self.robot_controller.relative_angle -= 2*pi

            self.robot_controller.desired_values[0] = x_obj + self.robot_controller.distance*cos(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[1] = y_obj + self.robot_controller.distance*sin(self.robot_controller.relative_angle)
            self.robot_controller.desired_values[3] = self.robot_controller.relative_angle +pi if self.robot_controller.relative_angle < 0 else self.robot_controller.relative_angle - pi

        elif action == "OPEN_1_1":
            if self.robot_controller.desired_values[5] < self.max_finger_angles and t < max_step:
                self.robot_controller.desired_values[5] +=   0.25/max_step
        elif action == "CLOSE_1_1":
            if t < max_step:
                self.robot_controller.desired_values[5] -=   0.25/max_step
        elif action == "OPEN_2_1":
            if self.robot_controller.desired_values[6] < self.max_finger_angles and t < max_step:
                self.robot_controller.desired_values[6] +=   0.25/max_step
        elif action == "CLOSE_2_1":
            if t < max_step:
                self.robot_controller.desired_values[6] -=   0.25/max_step
        elif action == "OPEN_1_2":
            if self.robot_controller.desired_values[7] < -0.0 and t < max_step:
                self.robot_controller.desired_values[7] +=   0.25/max_step
        elif action == "CLOSE_1_2":
            if t < max_step:
                self.robot_controller.desired_values[7] -=  0.25/max_step
        elif action == "OPEN_2_2":
            if self.robot_controller.desired_values[8] < -0.0 and t < max_step:
                self.robot_controller.desired_values[8] +=   0.25/max_step
        elif action == "CLOSE_2_2":
            if t < max_step:
                self.robot_controller.desired_values[8] -=  0.25/max_step

        elif action == "CHANGE_STATESPACE":
            self.robot_controller.desired_values[2] = 0.6
            self.robot_controller.desired_values[0] = x_obj*t/max_step
            self.robot_controller.desired_values[1] = y_obj*t/max_step
            self.robot_controller.desired_values[4] = (pi/2 - 0.01)*t/max_step
            self.statespace = 1

        elif action == "S2_ROT_RIGHT":
            if current_angle > -pi + 0.6:
                delta_phi = -0.6
            else:
                delta_phi = 0
            self.robot_controller.desired_values[3] = current_angle + delta_phi
        elif action == "S2_ROT_LEFT":
            if current_angle < pi - 0.6:
                delta_phi = 0.6
            else:
                delta_phi = 0
            self.robot_controller.desired_values[3] = current_angle+ delta_phi
        elif action == "S2_UP":
            if self.robot_controller.desired_values[2] < 0.6 and t < max_step:
                self.robot_controller.desired_values[2] += 0.1/max_step
        elif action == "S2_DOWN":
            if self.robot_controller.desired_values[2] > 0.35 and t < max_step:
                self.robot_controller.desired_values[2] -= 0.1/max_step
        elif action == "S2_FRONT":
            if (x_robot - x_obj) < 0.2:
                delta_x = 0.1*t/max_step
            else:
                delta_x = 0
            self.robot_controller.desired_values[0] = x_robot + delta_x
        elif action == "S2_BACK":
            if (x_robot - x_obj) > -0.2:
                delta_x = -0.1*t/max_step
            else:
                delta_x = 0
            self.robot_controller.desired_values[0] = x_robot + delta_x
        elif action == "S2_LEFT":
            if (y_obj - y_robot) > -0.2:
                delta_y = 0.1*t/max_step
            else:
                delta_y = 0
            self.robot_controller.desired_values[1] = y_robot + delta_y
        elif action == "S2_RIGHT":
            if (y_obj - y_robot) < 0.2:
                delta_y = -0.1*t/max_step
            else:
                delta_y = 0
            self.robot_controller.desired_values[1] = y_robot + delta_y
