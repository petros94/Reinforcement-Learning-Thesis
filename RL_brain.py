import sys
import numpy as np
import pandas as pd
from RL_environment import tilesPerTiling, numTilings, memsize
import bisect as bi

from RL_tests import *
from MPIconfig import rank, Single_Agent


"""
This code contains the most popular RL algorithms.
Warning: This file contains a lot of unfinished code. The following classes
can be used without any modification for Single or Multi Agents:

Q_learning_linear, expSARSA_linear

The following class can be used only on single agent environment:

SARSA_linear, Monte_Carlo

The other classes provide a good starting point, if you wish to apply
the algorithm to your problem.

"""
""" Helper classes """
class Queue():
    """fifo queue"""
    def __init__(self, n):
        self.queue = [[0, 0]]*n
        self.size = 0
        self.max_size = n

    def put(self,item):
        self.queue.pop(0)
        self.queue.append(item)
        if self.size < self.max_size:
            self.size+=1

    def empty(self):
        return 1 if self.size == 0 else 0

    def full(self):
        return 1 if self.size == self.max_size else 0

    def clear(self):
        self.size = 0

    def getall(self):
        return self.queue

    def __getitem__(self, index):
        return self.queue[index]


""" RL Classes """

""" Simple Tabular Q-Learning """

class Q_learning:

    def __init__(self, actions, learning_rate = 0.6, reward_decay = 0.95, e_greedy = 0.95):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.Q_table = pd.DataFrame(columns = self.actions, dtype = np.float32)

    def check_if_exists(self, state):
        if state not in self.Q_table.index:
            new_entry = pd.Series([0]*len(self.actions), index = self.Q_table.columns, name = state)
            self.Q_table = self.Q_table.append(new_entry)

    def choose_action(self, state):
        self.check_if_exists(state)
        state_action = self.Q_table.ix[state, :]
        prob = np.random.uniform()

        if prob < self.e:
            if state_action.unique().size != 1:
                action = state_action.idxmax()
            else:
                action = np.random.choice(self.actions)
                print("random")
        else:
            action = np.random.choice(self.actions)
            print("TOOTALY RANDOM")

        # print("values FRONT BACK: ", state_action.ix["FRONT"], state_action.ix["BACK"])
        return action

    def learn(self, s, a, r, s_):
        self.check_if_exists(s_)
        Q_value = self.Q_table.ix[s, a]

        if s_ == "terminal":
            Q_value = Q_value + self.lr*(r - Q_value)
        else:
            Q_max = self.Q_table.ix[s_,:].max()
            Q_value = Q_value + self.lr*(r + self.gamma*Q_max - Q_value)

        self.Q_table.ix[s, a] = Q_value


class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.set_value(s, a, (r, s_))


    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.ix[s].dropna().index)    # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.ix[s, a]
        return r, s_

""" SARSA with Linear Function Approximation (Tile Coding) """

class SARSA_linear:

    def __init__(self, action_list, learning_rate = 0.025, reward_decay = 0.9, e_greedy = 0.9):
        global numTilings
        self.action_list = action_list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.statespace = 0



        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = []
        self.experience_indices = []
        self.experience_values = []
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[0]), dtype = np.float32))
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[1]), dtype = np.float32))
        self.updated_weights = [[[0,0], 0, 1]]

    def choose_statespace(self, space):
        if space:
            action = "CHANGE_STATESPACE"
            self.statespace = 1
        else:
            action = "NONE"
        return action


    def choose_action(self, s):
        global numTilings
        feature_vector = s[0:numTilings]
        statespace = s[numTilings]

        Q = [0]*len(self.action_list[statespace])
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            for j in feature_vector:
                Q[idx] += self.weights[statespace][idx, j]

        Q_max_entry = np.argmax(Q)
        # action = self.action_list[statespace][Q_max_entry]

        actions_list = []
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            if Q[idx] == Q[Q_max_entry]:
                actions_list.append(action)
        action = np.random.choice(actions_list)

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            # print("random action taken")
            return np.random.choice(self.action_list[statespace]), np.max(Q)


        """For testing and measurements"""
        global rank
        if rank == 1:
            print("Q values: ", Q)
            print("Action taken: ", action)
            sys.stdout.flush()

        return action, np.max(Q)


    def learn(self, s, a, r, s_, a_, done):
        global numTilings
        statespace = s[numTilings]
        feature_vector = s[0:numTilings]
        statespace_ = s_[numTilings]
        feature_vector_ = s_[0:numTilings]

        idx = self.action_list[statespace].index(a)
        Q = 0
        for i in feature_vector:
            Q += self.weights[statespace][idx, i]

        idx_ = self.action_list[statespace_].index(a_)
        Q_ = 0
        for i in feature_vector_:
            Q_ += self.weights[statespace_][idx_, i]

        if done:
            for i in feature_vector:
                self.weights[statespace][idx, i] = self.weights[statespace][idx, i] + self.lr*(r - Q)
        else:
            for i in feature_vector:
                self.weights[statespace][idx,i] = self.weights[statespace][idx,i] + self.lr*(r + self.gamma*Q_ - Q)


""" Expected SARSA with Linear Function Approximation (Tile Coding) """

class expSARSA_linear:

    def __init__(self, action_list, learning_rate = 0.025, reward_decay = 0.9, e_greedy = 0.9):
        global numTilings
        self.action_list = action_list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.statespace = 0


        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = []
        self.experience_indices = []
        self.experience_values = []
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[0]), dtype = np.float32))
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[1]), dtype = np.float32))
        self.updated_weights = [[[0,0], 0, 1]]


    def reset(self):
        self.updated_weights = [[[0,0], 0, 1]]
        self.experience_indices = []
        self.experience_values = []


    def choose_statespace(self, space):
        if np.random.uniform() < space:
            action = "CHANGE_STATESPACE"
            self.statespace = 1
        else:
            action = "NONE"
        return action


    def choose_action(self, s):
        global numTilings
        feature_vector = s[0:numTilings]
        statespace = s[numTilings]

        Q = [0]*len(self.action_list[statespace])
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            for j in feature_vector:
                Q[idx] += self.weights[statespace][idx, j]

        # Q_max_entry = np.argmax(Q)
        # action = self.action_list[statespace][Q_max_entry]

        actions_list = []
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            if Q[idx] == np.max(Q):
                actions_list.append(action)
        action = np.random.choice(actions_list)

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            # print("random action taken")
            return np.random.choice(self.action_list[statespace]), np.max(Q)


        """For testing and measurements"""
        global rank
        if rank == 1:
            print("Q values: ", Q)
            print("Action taken: ", action)
            sys.stdout.flush()

        return action, np.max(Q)


    def learn(self, s, a, r, s_, done):
        global numTilings
        statespace = s[numTilings]
        feature_vector = s[0:numTilings]

        idx = self.action_list[statespace].index(a)
        Q = 0
        for i in feature_vector:
            Q += self.weights[statespace][idx, i]

        statespace_ = s_[numTilings]
        feature_vector_ = s_[0:numTilings]
        Q_ = [0]*len(self.action_list[statespace_])
        for action in self.action_list[statespace_]:
            idx_ = self.action_list[statespace_].index(action)
            for j in feature_vector_:
                Q_[idx_] += self.weights[statespace_][idx_, j]

        expected_Q = self.e*np.max(Q_) + (1-self.e)/len(self.action_list[statespace_])*np.sum(Q_)

        """Update Q values"""
        if done:
            for i in feature_vector:
                self.weights[statespace][idx, i] = self.weights[statespace][idx, i] + self.lr*(r - Q)

        else:
            for i in feature_vector:
                self.weights[statespace][idx,i] = self.weights[statespace][idx,i] + self.lr*(r + self.gamma*expected_Q - Q)


        """This segment is responsible for gathering the weights, that will be sent to the main node, in the case of parallel learning"""
        for i in feature_vector:
            try:
                exp_idx = self.experience_indices.index([i,idx])
                self.experience_values[exp_idx] +=1
                self.updated_weights[exp_idx] = [[i,idx], self.weights[statespace][idx,i], self.experience_values[exp_idx]]
            except ValueError:
                self.experience_indices.append([i,idx])
                self.experience_values.append(1)
                self.updated_weights.append([[i,idx], self.weights[statespace][idx,i], 1])



    def avg_learn(self, statespace, s_idx, a_idx, avg_weight, exp_factor):
        self.weights[statespace][a_idx, s_idx] = avg_weight


""" Q-Learning with Linear Function Approximation """

class Q_learning_linear:

    def __init__(self, actions, learning_rate = 0.025, reward_decay = 0.9, e_greedy = 0.9):
        global numTilings
        self.action_list = action_list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.statespace = 0

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = []
        self.experience_indices = []
        self.experience_values = []
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[0]), dtype = np.float32))
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[1]), dtype = np.float32))
        self.updated_weights = [[[0,0], 0, 1]]

    def reset(self):
        self.updated_weights = [[[0,0], 0, 1]]
        self.experience_indices = []
        self.experience_values = []


    def choose_statespace(self, space):
        if np.random.uniform() < space:
            action = "CHANGE_STATESPACE"
            self.statespace = 1
        else:
            action = "NONE"
        return action


    def choose_action(self, s):
        global numTilings
        feature_vector = s[0:numTilings]
        statespace = s[numTilings]

        Q = [0]*len(self.action_list[statespace])
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            for j in feature_vector:
                Q[idx] += self.weights[statespace][idx, j]


        actions_list = []
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            if Q[idx] == np.max(Q):
                actions_list.append(action)
        action = np.random.choice(actions_list)

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            # print("random action taken")
            return np.random.choice(self.action_list[statespace]), np.max(Q)


        """For testing and measurements"""
        global rank
        if rank == 1:
            print("Q values: ", Q)
            print("Action taken: ", action)
            sys.stdout.flush()

        return action, np.max(Q)


    def learn(self, s, a, r, s_, done):
        global numTilings
        statespace = s[numTilings]
        feature_vector = s[0:numTilings]


        idx = self.action_list[statespace].index(a)
        Q = 0
        for i in feature_vector:
            Q += self.weights[statespace][idx, i]

        statespace_ = s_[numTilings]
        feature_vector_ = s_[0:numTilings]
        Q_ = [0]*len(self.actions)
        for action in self.action_list[statespace_]:
            idx_ = self.action_list[statespace_].index(action)
            for j in feature_vector_:
                Q_[idx_] += self.weights[statespace_][idx_, j]


        """Update Q values"""
        if done:
            for i in feature_vector:
                self.weights[statespace][idx, i] = self.weights[statespace][idx, i] + self.lr*(r - Q)

        else:
            for i in feature_vector:
                self.weights[statespace][idx,i] = self.weights[statespace][idx,i] + self.lr*(r + self.gamma*np.max(Q_) - Q)

        """This segment is responsible for gathering the weights, that will be sent to the main node, in the case of parallel learning"""
        for i in feature_vector:
            try:
                exp_idx = self.experience_indices.index([i,idx])
                self.experience_values[exp_idx] +=1
                self.updated_weights[exp_idx] = [[i,idx], self.weights[statespace][idx,i], self.experience_values[exp_idx]]
            except ValueError:
                self.experience_indices.append([i,idx])
                self.experience_values.append(1)
                self.updated_weights.append([[i,idx], self.weights[statespace][idx,i], 1])

    def avg_learn(self, statespace, s_idx, a_idx, avg_weight, exp_factor):
        self.weights[statespace][a_idx, s_idx] = avg_weight


""" Q-Learning with eligibility trace and linear function approximation """

class Q_lambda_linear:

    def __init__(self, actions, learning_rate = 0.05, reward_decay = 0.95, e_greedy = 0.9):
        global numTilings
        global memsize
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.eligibility_vector = [[[0, 0], 0]]
        self.l = 0.4

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)
        self.updated_weights = [[[0,0], 0]]

    def reset(self):
        self.updated_weights = [[[0,0], 0]]
        self.eligibility_vector = [[[0, 0], 0]]


    def choose_action(self, s):
        self.feature_vector = np.array(s)

        Q = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in self.feature_vector:
                Q[idx] += self.weights[idx, j]

        Q_max_entry = np.argmax(Q)

        action_list = []
        for action in self.actions:
            idx = self.actions.index(action)
            if Q[idx] == np.max(Q):
                action_list.append(action)
        action = np.random.choice(action_list)


        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            print("Random action chosen")
            action = np.random.choice(self.actions)


        """For testing and measurements"""
        global rank
        if rank == 1:
            print("Q values: ", Q)
            print("Action taken: ", action)
            sys.stdout.flush()

        return action


    def learn(self, s, a, r, s_, done):
        idx = self.actions.index(a)

        """Update eligibility vector"""
        for entry in self.eligibility_vector:
            entry[1] *= self.gamma * self.l

        """add to eligibility vector"""
        eligibility_indices, eligibility_values = zip(*self.eligibility_vector)
        for i in s:
            index = bi.bisect_left(eligibility_indices, [idx, i])
            if index != len(eligibility_indices) and eligibility_indices[index] == [idx,i]:
                self.eligibility_vector[index] = [[idx, i], 1]
            else:
                self.eligibility_vector.insert(index, [[idx, i], 1])

        """Calculate Q values"""
        Q = 0
        for i in s:
            Q += self.weights[idx, i]

        Q_ = [0]*len(self.actions)
        for action in self.actions:
            idx_ = self.actions.index(action)
            for j in s_:
                Q_[idx_] += self.weights[idx_, j]

        """Update weights using eligibility vector"""
        if done:
            indices, values = zip(*self.updated_weights)
            for entry in self.eligibility_vector:
                idx = entry[0][0]
                i = entry[0][1]
                eligibility_value = entry[1]
                self.weights[idx, i] = self.weights[idx, i] + self.lr*(r - Q)*eligibility_value

                """MPI - add to updated weights"""
                index = bi.bisect_left(indices, [idx, i])
                if index != len(indices) and indices[index] == [idx,i]:
                    self.updated_weights[index] = [[idx, i], self.weights[idx, i]]
                else:
                    self.updated_weights.insert(index, [[idx, i], self.weights[idx, i]])
        else:
            indices, values = zip(*self.updated_weights)
            for entry in self.eligibility_vector:
                idx = entry[0][0]
                i = entry[0][1]
                eligibility_value = entry[1]
                self.weights[idx,i] = self.weights[idx,i] + self.lr*(r + self.gamma*np.max(Q_) - Q)*eligibility_value

                """MPI - add to updated weights"""
                index = bi.bisect_left(indices, [idx, i])
                if index != len(indices) and indices[index] == [idx,i]:
                    self.updated_weights[index] = [[idx, i], self.weights[idx, i]]
                else:
                    self.updated_weights.insert(index, [[idx, i], self.weights[idx, i]])


""" DQN code from github """

import sklearn.neural_network as sk

class Deep_Q_learning:

    def __init__(self, actions, learning_rate = 0.001, reward_decay = 0.95, e_greedy = 0.95):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.n_features = 3

        self.Qnet = sk.MLPRegressor(learning_rate_init = learning_rate, hidden_layer_sizes = (18), activation = "relu", solver = "adam")
        self.Qnet.n_outputs = len(self.actions)
        self.Qnet.max_iter = 20
        self.batch_size = 100

        self.memory_size = 0
        self.memory_max_size = 500
        self.memory = np.zeros((self.memory_max_size, (2*self.n_features + 3)))

        init_q_target = np.array([-1]*self.batch_size*4)
        init_q_target.shape = [self.batch_size, 4]
        init_q_target = np.array([-1]*8)
        init_q_target.shape = [2, 4]
        self.Qnet.fit([[0, 0, 0], [0, 0, 0]], init_q_target)

    def init_q(self, s):

        init_q_target = np.array([-1]*8)
        init_q_target.shape = [2, 4]
        self.Qnet.fit([s, s], init_q_target)


    def choose_action(self, s):

        if np.random.uniform() > self.e:
            return np.random.choice(self.actions)

        Q_values = self.Qnet.predict([s])
        # print(Q_values)
        idx = np.argmax(np.array(Q_values))
        action = self.actions[idx]
        return action

    def store_transition(self, s, a, r, s_, a_):
        transition = np.array([s, [int(self.actions.index(a)),r, int(self.actions.index(a_))], s_])
        transition = np.hstack(transition)
        idx = self.memory_size
        if self.memory_size >= self.memory_max_size:
            idx = self.memory_size % self.memory_max_size

        self.memory[idx,:] = transition
        self.memory_size += 1

    def learn(self):
        """pick random samples"""
        indices = 0
        if self.memory_size < self.memory_max_size:
            indices = np.random.choice(self.memory_size, self.batch_size)
        else:
            indices = np.random.choice(self.memory_max_size, self.batch_size)

        # print(indices)

        batch = self.memory[indices, :]

        """predict Q and Q_ values"""
        network_input_s = batch[:, :self.n_features]
        network_input_s_ = batch[:, -self.n_features:]

        actions = batch[:, self.n_features]
        actions_ = batch[:, self.n_features+2]
        rewards = batch[:, self.n_features+1]

        Q = self.Qnet.predict(network_input_s)
        Q_ = self.Qnet.predict(network_input_s_)

        """Set target for training"""
        target = Q.copy()
        for i in range(self.batch_size):
            action_taken = actions[i]
            action_taken_ = actions_[i]
            target[i, int(action_taken)] = rewards[i] + self.gamma*Q_[i,int(action_taken_)]

        """train the network"""
        self.Qnet.fit(network_input_s, target)


class Actor_linear:
    def __init__(self, actions, learning_rate = 0.05, reward_decay = 0.95, e_greedy = 0.95):
        global numTilings
        global memsize
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)

    def choose_action(self, s):
        H = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in s:
                H[idx] += self.weights[idx, j]

        p = np.exp(np.array(H))/np.sum(np.exp(np.array(H)))
        print("Propabilities: ", p)
        sys.stdout.flush()
        random_number = np.random.uniform()
        sum = 0
        if np.random.uniform() > self.e:
            return np.random.choice(self.actions)

        for i in range(len(self.actions)):
            sum += p[i]
            if random_number <= sum:
                action = self.actions[i]
                break

        return action

    def learn(self, s, a, delta):
        H = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in s:
                H[idx] += self.weights[idx, j]

        p = np.exp(np.array(H))/np.sum(np.exp(np.array(H)))
        idx = self.actions.index(a)

        for action in self.actions_list:
            idx = self.actions.index(action)
            if action == a:
                for i in s:
                    self.weights[statespace][idx, i] += self.lr*delta*(1 - p[idx])




class Critic_linear:
    def __init__(self, actions, learning_rate = 0.02, reward_decay = 0.95, e_greedy = 0.95):
        global numTilings
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)

    def learn(self, s, a, r, s_, a_, done):

        idx = self.actions.index(a)
        Q = 0
        for i in s:
            Q += self.weights[idx, i]

        idx_ = self.actions.index(a_)
        Q_ = 0
        for i in s_:
            Q_ += self.weights[idx_, i]

        delta = 0
        if done:
            for i in s:
                self.weights[idx, i] = self.weights[idx, i] + self.lr*(r - Q)
                delta = (r - Q)
        else:
            for i in s:
                self.weights[idx,i] = self.weights[idx,i] + self.lr*(r + self.gamma*Q_ - Q)
                delta = r + self.gamma*Q_ - Q

        print("Delta is: ", delta)
        sys.stdout.flush()
        return delta

class n_step_SARSA_linear:

    def __init__(self, actions, learning_rate = 0.03, reward_decay = 0.95, e_greedy = 0.9, step_size = 0):
        global tilesPerTiling
        global numTilings
        global memsize
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.n = step_size

        self.reward_fifo = Queue(self.n+1)
        self.state_action_indices_fifo = Queue(self.n+1)
        self.gamma_fifo = Queue(self.n+1)

        """init gamma fifo"""
        for i in range(self.n+1):
            g = pow(self.gamma, i)
            self.gamma_fifo.put(g)

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)


    def choose_action(self, s):
        self.feature_vector = np.array(s)

        Q = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in self.feature_vector:
                Q[idx] += self.weights[idx, j]

        Q_max_entry = np.argmax(Q)
        action = self.actions[Q_max_entry]

        # if Q[Q_max_entry] == 0:
        #     random_action_list = []
        #     for action in self.actions:
        #         idx = self.actions.index(action)
        #         if Q[idx] == 0:
        #             random_action_list.append(action)
        #     action = np.random.choice(random_action_list)
        #     print("random action taken")

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            print("random action taken")
            return np.random.choice(self.actions)


        """For testing and measurements"""
        # write_to_file(0, Q, action, 0)
        print("Q values of actions: ", Q)
        print("Next action: ", action)

        return action


    def learn(self, s, a, r, s_, a_, done):

        """store [s,a], r"""
        idx = self.actions.index(a)
        self.state_action_indices_fifo.put([s, idx])
        self.reward_fifo.put(r)

        print("Rewards: ", self.reward_fifo.getall())
        b=[]
        for entry in self.state_action_indices_fifo:
            b.append(self.actions[entry[1]])
        print("Past Actions: ", b)

        """Measure state similarity"""
        counter = 0
        for i in s:
            for j in s_:
                if i == j:
                    counter += 1
        print("Similarity with previous state: ", float(counter)/16*100, "%")


        """if queue is full, update first value"""
        if self.state_action_indices_fifo.full():

            """If done update all Q_values"""
            if done:
                print("Update is in terminal")
                for i in range(self.n+1):
                    Gt = 0
                    for j in range(self.n+1-i):
                        Gt += pow(self.gamma, j)*self.reward_fifo[i+j]

                    s0, a0 = self.state_action_indices_fifo[i]
                    Q0 = 0
                    for k in s0:
                        Q0 += self.weights[a0, k]

                    """Update weights"""
                    for k in s0:
                        self.weights[a0, k] += self.lr*(Gt - Q0)

                """Empty queue after end of episode"""
                self.reward_fifo.clear()
                self.state_action_indices_fifo.clear()

            else:
                """Else update only one value"""
                print("Normal update")
                s0, a0 = self.state_action_indices_fifo[0]
                Q0 = 0
                for i in s0:
                    Q0 += self.weights[a0, i]

                Gt = 0
                for i in range(self.n+1):
                    g = pow(self.gamma, i)
                    Gt += g*self.reward_fifo[i]

                idx_ = self.actions.index(a_)
                Q_ = 0
                for i in s_:
                    Q_ += self.weights[idx_, i]
                Gt += pow(self.gamma, self.n+1)*Q_

                """Finally update weights"""
                for i in s0:
                    self.weights[a0, i] = self.weights[a0, i] + self.lr*(Gt - Q0)

    def change_step(self, step_size):
        self.n = step_size
        self.reward_fifo = Queue(self.n+1)
        self.state_action_indices_fifo = Queue(self.n+1)
        self.gamma_fifo = Queue(self.n+1)


class n_step_expSARSA_linear:

    def __init__(self, actions, learning_rate = 0.06, reward_decay = 0.95, e_greedy = 0.9, step_size = 0):
        global tilesPerTiling
        global numTilings
        global memsize
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.n = step_size

        self.reward_fifo = Queue(self.n+1)
        self.state_action_indices_fifo = Queue(self.n+1)

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)

        self.total_reward = 0
        self.t = 0
        plt.clf()
        plt.axis([0, 200, -100, 100])
        plt.ion()



    def choose_action(self, s):
        self.feature_vector = np.array(s)

        Q = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in self.feature_vector:
                Q[idx] += self.weights[idx, j]

        Q_max_entry = np.argmax(Q)
        action = self.actions[Q_max_entry]

        action_list = []
        for action in self.actions:
            idx = self.actions.index(action)
            if Q[idx] == np.max(Q):
                action_list.append(action)
        action = np.random.choice(action_list)

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            print("random action taken")
            return np.random.choice(self.actions)


        """For testing and measurements"""
        # write_to_file(0, Q, action, 0)
        print("Q values of actions: ", Q)
        print("Next action: ", action)

        return action


    def learn(self, s, a, r, s_,a_, done):

        """store [s,a], r"""
        idx = self.actions.index(a)
        self.state_action_indices_fifo.put([s, idx])
        self.reward_fifo.put(r)

        print("Rewards: ", self.reward_fifo.getall())
        b=[]
        for entry in self.state_action_indices_fifo:
            b.append(self.actions[entry[1]])
        print("Past Actions: ", b)

        """Measure state similarity"""
        counter = 0
        for i in s:
            for j in s_:
                if i == j:
                    counter += 1
        print("Similarity with previous state: ", float(counter)/16*100, "%")


        """If done update all Q_values"""
        if done:
            print("Update is in terminal")
            for i in range(self.n - self.reward_fifo.size + 1, self.n+1):
                Gt = 0
                for j in range(self.n+1-i):
                    Gt += pow(self.gamma, j)*self.reward_fifo[i+j]

                s0, a0 = self.state_action_indices_fifo[i]
                Q0 = 0
                for k in s0:
                    Q0 += self.weights[a0, k]

                """Update weights"""
                for k in s0:
                    self.weights[a0, k] += self.lr*(Gt - Q0)

            """Empty queue after end of episode"""
            self.reward_fifo.clear()
            self.state_action_indices_fifo.clear()


        else:
            """if queue is full, update first value"""
            if self.state_action_indices_fifo.full():

                print("Normal update")
                s0, a0 = self.state_action_indices_fifo[0]
                Q0 = 0
                for i in s0:
                    Q0 += self.weights[a0, i]

                Gt = 0
                for i in range(self.n+1):
                    g = pow(self.gamma, i)
                    Gt += g*self.reward_fifo[i]

                Q_ = [0]*len(self.actions)
                for action in self.actions:
                    idx_ = self.actions.index(action)
                    for j in s_:
                        Q_[idx_] += self.weights[idx_, j]


                expected_Q = self.e*np.max(Q_) + (1-self.e)/len(self.actions)*np.sum(Q_)

                Gt += pow(self.gamma, self.n+1)*expected_Q

                """Finally update weights"""
                for i in s0:
                    self.weights[a0, i] = self.weights[a0, i] + self.lr*(Gt - Q0)

    def change_step(self, step_size):
        self.n = step_size
        self.reward_fifo = Queue(self.n+1)
        self.state_action_indices_fifo = Queue(self.n+1)
        self.gamma_fifo = Queue(self.n+1)

class n_step_Q_linear:

    def __init__(self, actions, learning_rate = 0.03, reward_decay = 0.95, e_greedy = 0.9, step_size = 0):
        global tilesPerTiling
        global numTilings
        global memsize
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.n = step_size

        self.reward_fifo = Queue(self.n+1)
        self.state_action_indices_fifo = Queue(self.n+1)
        self.gamma_fifo = Queue(self.n+1)

        """init gamma fifo"""
        for i in range(self.n+1):
            g = pow(self.gamma, i)
            self.gamma_fifo.put(g)

        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = np.array([[0.1]*len(self.feature_vector)]*len(actions), dtype = np.float32)


    def choose_action(self, s):
        self.feature_vector = np.array(s)

        Q = [0]*len(self.actions)
        for action in self.actions:
            idx = self.actions.index(action)
            for j in self.feature_vector:
                Q[idx] += self.weights[idx, j]

        Q_max_entry = np.argmax(Q)
        action = self.actions[Q_max_entry]

        if Q[Q_max_entry] == 0:
            random_action_list = []
            for action in self.actions:
                idx = self.actions.index(action)
                if Q[idx] == 0:
                    random_action_list.append(action)
            action = np.random.choice(random_action_list)
            print("random action taken")

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            print("random action taken")
            return np.random.choice(self.actions)


        """For testing and measurements"""
        # write_to_file(0, Q, action, 0)
        print("Q values of actions: ", Q)
        print("Next action: ", action)

        return action


    def learn(self, s, a, r, s_, done):

        """store [s,a], r"""
        idx = self.actions.index(a)
        self.state_action_indices_fifo.put([s, idx])
        self.reward_fifo.put(r)

        print("Rewards: ", self.reward_fifo.getall())
        b=[]
        for entry in self.state_action_indices_fifo:
            b.append(self.actions[entry[1]])
        print("Past Actions: ", b)


        """if queue is full, update first value"""
        if self.state_action_indices_fifo.full():

            """If done update all Q_values"""
            if done:
                print("Update is in terminal")
                for i in range(self.n+1):
                    Gt = 0
                    for j in range(self.n+1-i):
                        Gt += pow(self.gamma, j)*self.reward_fifo[i+j]

                    s0, a0 = self.state_action_indices_fifo[i]
                    Q0 = 0
                    for k in s0:
                        Q0 += self.weights[a0, k]

                    """Update weights"""
                    for k in s0:
                        self.weights[a0, k] += self.lr*(Gt - Q0)

                """Empty queue after end of episode"""
                self.reward_fifo.clear()
                self.state_action_indices_fifo.clear()

            else:
                """Else update only one value"""
                print("Normal update")
                s0, a0 = self.state_action_indices_fifo[0]
                Q0 = 0
                for i in s0:
                    Q0 += self.weights[a0, i]

                Gt = 0
                for i in range(self.n+1):
                    g = pow(self.gamma, i)
                    Gt += g*self.reward_fifo[i]

                Q_ = [0]*len(self.actions)
                for action in self.actions:
                    idx_ = self.actions.index(action)
                    for j in s_:
                        Q_[idx_] += self.weights[idx_, j]

                Gt += pow(self.gamma, self.n+1)*np.max(Q_)

                """Finally update weights"""
                for i in s0:
                    self.weights[a0, i] = self.weights[a0, i] + self.lr*(Gt - Q0)

class Monte_Carlo:
    def __init__(self, action_list, learning_rate = 0.025, reward_decay = 0.9, e_greedy = 0.9):
        global memsize
        self.action_list = action_list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.e = e_greedy
        self.t = 0
        self.statespace = 0

        self.reward_list = []
        self.state_action_indices_list = []
        self.returns = []
        self.returns_indices = []


        self.feature_vector = np.array([0]*memsize, dtype = np.float32)
        self.weights = []
        self.experience_indices = []
        self.experience_values = []
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[0]), dtype = np.float32))
        self.weights.append(np.array([[0.0]*len(self.feature_vector)]*len(action_list[1]), dtype = np.float32))

    def choose_statespace(self, space):
        if np.random.uniform() < space:
            action = "CHANGE_STATESPACE"
            self.statespace = 1
        else:
            action = "NONE"
        return action


    def choose_action(self, s):
        feature_vector = s[0:numTilings]
        statespace = s[numTilings]

        Q = [0]*len(self.action_list[statespace])
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            for j in feature_vector:
                Q[idx] += self.weights[statespace][idx, j]

        Q_max_entry = np.argmax(Q)
        # action = self.actions[Q_max_entry]

        actions_list = []
        for action in self.action_list[statespace]:
            idx = self.action_list[statespace].index(action)
            if Q[idx] == Q[Q_max_entry]:
                actions_list.append(action)
        action = np.random.choice(actions_list)

        """ e-greedy policy"""
        if np.random.uniform() > self.e:
            # print("random action taken")
            return np.random.choice(self.action_list[statespace]), np.max(Q)


        """For testing and measurements"""
        # write_to_file(0, Q, action, 0)
        print("Q values of actions: ", Q)
        print("Next action: ", action)


        return action, np.max(Q)

    def learn(self, s, a, r, s_, done):
        feature_vector = s[0:numTilings]
        statespace = s[numTilings]

        """store [s,a], r"""
        idx = self.action_list[statespace].index(a)
        if [feature_vector, idx] not in self.state_action_indices_list:
            self.state_action_indices_list.append([feature_vector, idx])
        else:
            self.state_action_indices_list.append([[0]*16, idx])
        self.reward_list.append(r)



        """Diagnostics"""
        print("Rewards size: ", len(self.reward_list))
        # b=[]
        # for entry in self.state_action_indices_list:
        #     b.append(self.actions[entry[1]])
        # print("Past Actions: ", b)


        """If done update all Q_values"""
        if done:

            print("Update is in terminal")
            for i in range(len(self.reward_list)):
                Gt = 0
                for j in range(len(self.reward_list)-i):
                    Gt += pow(self.gamma, j)*self.reward_list[i+j]

                s0, a0 = self.state_action_indices_list[i]

                ret_idx = bi.bisect_left(self.returns_indices, [s0, a0])
                if ret_idx != len(self.returns_indices) and self.returns_indices[ret_idx] == [s0, a0]:
                    """ state-action found in history: increment by one"""
                    self.returns[ret_idx].append(Gt)
                else:
                    self.returns_indices.insert(ret_idx, [s0, a0])
                    self.returns.insert(ret_idx, [Gt])

                Q0 = 0
                for k in s0:
                    Q0 += self.weights[statespace][a0, k]

                avg_Gt = np.mean(self.returns[ret_idx])

                """Update weights"""
                for k in s0:
                    self.weights[statespace][a0, k] += self.lr*(avg_Gt - Q0)

            """Empty queue after end of episode"""
            self.reward_list = []
            self.state_action_indices_list = []

# import tensorflow as tf
#
# class DeepQNetwork:
#     def __init__(
#             self,
#             actions,
#             n_features,
#             learning_rate=0.001,
#             reward_decay=0.95,
#             e_greedy=0.9,
#             replace_target_iter=16,
#             memory_size=32,
#             batch_size=8,
#             e_greedy_increment=None,
#             output_graph=False,
#     ):
#         self.actions = actions
#         self.n_actions = len(actions)
#         self.n_features = n_features
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon_max = e_greedy
#         self.replace_target_iter = replace_target_iter
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.epsilon_increment = e_greedy_increment
#         self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
#
#         # total learning step
#         self.learn_step_counter = 0
#
#         # initialize zero memory [s, a, r, s_]
#         self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
#
#         # consist of [target_net, evaluate_net]
#         self._build_net()
#         t_params = tf.get_collection('target_net_params')
#         e_params = tf.get_collection('eval_net_params')
#         self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
#
#         self.sess = tf.Session()
#
#         if output_graph:
#             # $ tensorboard --logdir=logs
#             # tf.train.SummaryWriter soon be deprecated, use following
#             tf.summary.FileWriter("logs/", self.sess.graph)
#
#         self.sess.run(tf.global_variables_initializer())
#         self.cost_his = []
#
#     def _build_net(self):
#         # ------------------ build evaluate_net ------------------
#         self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
#         self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
#         with tf.variable_scope('eval_net'):
#             # c_names(collections_names) are the collections to store variables
#             c_names, n_l1, w_initializer, b_initializer = \
#                 ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 40, \
#                 tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
#
#             # first layer. collections is used later when assign to target net
#             with tf.variable_scope('l1'):
#                 w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
#                 b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
#                 l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
#
#             # second layer. collections is used later when assign to target net
#             with tf.variable_scope('l2'):
#                 w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
#                 b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                 self.q_eval = tf.matmul(l1, w2) + b2
#
#         with tf.variable_scope('loss'):
#             self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
#         with tf.variable_scope('train'):
#             self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
#
#         # ------------------ build target_net ------------------
#         self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
#         with tf.variable_scope('target_net'):
#             # c_names(collections_names) are the collections to store variables
#             c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
#
#             # first layer. collections is used later when assign to target net
#             with tf.variable_scope('l1'):
#                 w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
#                 b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
#                 l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
#
#             # second layer. collections is used later when assign to target net
#             with tf.variable_scope('l2'):
#                 w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
#                 b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                 self.q_next = tf.matmul(l1, w2) + b2
#
#     def store_transition(self, s, a, r, s_):
#         if not hasattr(self, 'memory_counter'):
#             self.memory_counter = 0
#
#         action = self.actions.index(a)
#         transition = np.hstack((s, [action, r], s_))
#
#         # replace the old memory with new memory
#         index = self.memory_counter % self.memory_size
#         self.memory[index, :] = transition
#
#         self.memory_counter += 1
#
#     def choose_action(self, observation):
#         # to have batch dimension when feed into tf placeholder
#         observation = observation[np.newaxis, :]
#
#         if np.random.uniform() < self.epsilon:
#             # forward feed the observation and get q value for every actions
#             actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
#             print("Q VALUES: ", actions_value)
#             action = np.argmax(actions_value)
#         else:
#             action = np.random.randint(0, self.n_actions)
#
#         print("Action taken: ", self.actions[action])
#         return self.actions[action]
#
#     def learn(self):
#         # check to replace target parameters
#         if self.learn_step_counter % self.replace_target_iter == 0:
#             self.sess.run(self.replace_target_op)
#             print('\ntarget_params_replaced\n')
#
#         # sample batch memory from all memory
#         if self.memory_counter > self.memory_size:
#             sample_index = np.random.choice(self.memory_size, size=self.batch_size)
#         else:
#             sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
#         batch_memory = self.memory[sample_index, :]
#
#         q_next, q_eval = self.sess.run(
#             [self.q_next, self.q_eval],
#             feed_dict={
#                 self.s_: batch_memory[:, -self.n_features:],  # fixed params
#                 self.s: batch_memory[:, :self.n_features],  # newest params
#             })
#
#         # change q_target w.r.t q_eval's action
#         q_target = q_eval.copy()
#
#         batch_index = np.arange(self.batch_size, dtype=np.int32)
#         eval_act_index = batch_memory[:, self.n_features].astype(int)
#         reward = batch_memory[:, self.n_features + 1]
#
#         q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
#
#         """
#         For example in this batch I have 2 samples and 3 actions:
#         q_eval =
#         [[1, 2, 3],
#          [4, 5, 6]]
#
#         q_target = q_eval =
#         [[1, 2, 3],
#          [4, 5, 6]]
#
#         Then change q_target with the real q_target value w.r.t the q_eval's action.
#         For example in:
#             sample 0, I took action 0, and the max q_target value is -1;
#             sample 1, I took action 2, and the max q_target value is -2:
#         q_target =
#         [[-1, 2, 3],
#          [4, 5, -2]]
#
#         So the (q_target - q_eval) becomes:
#         [[(-1)-(1), 0, 0],
#          [0, 0, (-2)-(6)]]
#
#         We then backpropagate this error w.r.t the corresponding action to network,
#         leave other action as error=0 cause we didn't choose it.
#         """
#
#         # train eval network
#         _, self.cost = self.sess.run([self._train_op, self.loss],
#                                      feed_dict={self.s: batch_memory[:, :self.n_features],
#                                                 self.q_target: q_target})
#         self.cost_his.append(self.cost)
#
#         # increasing epsilon
#         self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
#         self.learn_step_counter += 1
