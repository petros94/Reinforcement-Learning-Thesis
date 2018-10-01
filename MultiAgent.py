import os
import math
import sys, getopt
from MPIconfig import *

"""Hub node"""
if rank == 0:


    import numpy as np

    if comm.size < 2:
        sys.exit()

    import bisect as bi


    while True:
        state_action_indices = []
        experience_values = []
        weights = []

        counter = 0
        """Wait for node transmission"""
        data_in = [0,0,0,0]
        data_out = [[], [], [], []]
        update_indices = []
        for node in range(1, comm.size):
            data_in[node] = comm.recv(source = node, tag = 0)
            print("HUB: data received from node ", node)
            sys.stdout.flush()

        """All data received - proccess data"""
        for node in range(1, comm.size):
            for entry in data_in[node]:
                try:
                    idx = state_action_indices.index(entry[0])
                    weights[idx][node] = entry[1]
                    experience_values[idx][node] = entry[2]
                    counter += 1
                except ValueError:
                    state_action_indices.append(entry[0])
                    new_weights = [0]*comm.size
                    new_experience = [0]*comm.size
                    new_weights[node] = entry[1]
                    new_experience[node] = entry[2]
                    weights.append(new_weights)
                    experience_values.append(new_experience)


        print(counter, " same states, total: ",  len(state_action_indices))


        """Calculate weighted average"""
        for idx in range(len(state_action_indices)):
            average = 0
            for node in range(1, comm.size):
                average += weights[idx][node]*experience_values[idx][node]

            average/= np.sum(experience_values[idx])

            """Write to output buffer"""
            for node in range(1, comm.size):
                weights[idx][node] = average
                data_out[node].append([state_action_indices[idx], weights[idx][node], 0])

        """Send to nodes"""
        for node in range(1, comm.size):
            print("Data set to process: ", node)
            sys.stdout.flush()
            comm.send(data_out[node], dest = node, tag = 0)

else:

    from RL_brain import Q_learning, EnvModel, SARSA_linear, Q_learning_linear, Actor_linear, Critic_linear, n_step_SARSA_linear, n_step_Q_linear, n_step_expSARSA_linear, Monte_Carlo, expSARSA_linear, Q_lambda_linear
    from RL_environment import *
    from RL_tests import *


    """Iniitalize environment"""
    env = Robot()


    """
    run_simuation

    Description:
    This function runs a simulation for a certain number of episodes.

    Arguments:
    agent - instance of an agent class. Can be any class defined in RL_brain.py
    statespace - can be 0 or 1. 0 means operating in statespace "S_side" and 1 in statespace "S_top". See Mitseas thesis, chapter 3.
    n_episodes - number of episodes that the simulation will run
    convex_option - can be 0, 1, or 2. Specifies the evaluation method used for grasping (and the reward function). 0 means evaluation by Inner Product,
                    1 means evaluation by size of Convex Hull, and 2 means that the reward function won't use grasp evaluation
    random_position - Setting this to 1, initializes the object with a random orientation at the start of each episode.
    export_path - export path for the weights that the agent learns thoughout the process
    export_flag - Set this to 1 if you wish to export the weights (q function) learned by the agent. Else set this to 0.
    import_path - import path for the initial q function weights
    import_flag - Set this to 1 if you wish to import initial q function weights. Else set to 0
    comm_frequency - select how often will the agents exchange data (number of episodes)

    Note: The imported weights must have the same size and format as the exported weights.
    """

    def run_simulation(agent, statespace, n_episodes, convex_option, random_position, export_path, export_flag, import_path, import_flag, comm_frequency):
        global rank

        """Set convex option"""
        env.path_planner.robot_controller.convex_option = convex_option


        """Variable initializations"""
        np.random.seed()
        t = 0
        success = "NO"
        episode = 0

        """Import/Export weights"""
        if import_flag:
            import_linear_weights(import_path, agent, statespace)


        """Main loop"""
        while episode < n_episodes:

            s = env.reset(0, random_position)
            a = agent.choose_statespace(statespace)
            s_, r, done = env.step(a, 0)
            s = s_

            while True:
                if rank == 1:
                    print("EPISODE: ", episode)
                    print("Last episode successful: ", success)

                a, maxQ = agent.choose_action(s)
                s_, r, done = env.step(a, 0)
                agent.learn(s, a, r, s_, done)

                s = s_

                if rank == 1:
                    print("-------------------------")

                if done:
                    break

            success = "YES" if r > 0 else "NO"
            episode += 1

            """""""""""""""""""MPI Broadcast"""""""""""""""""""""
            if episode % comm_frequency == 0:
                data_out = agent.updated_weights
                """Send to hub"""
                comm.send(data_out, dest = 0, tag = 0)
                """Wait for answer"""
                global memsize
                print("PROCESS ", rank, ": waiting for data...")
                sys.stdout.flush()
                data_in = comm.recv(source = 0, tag = 0)
                print("Rank: ", rank, "sent ", len(data_out), "state-asctons")
                print("PROCESS ", rank, ": Data received ")
                sys.stdout.flush()
                for entry in data_in:
                    s_idx = entry[0][0]
                    a_idx = entry[0][1]
                    average_weight = entry[1]
                    exp_factor = entry[2]
                    agent.avg_learn(statespace, s_idx, a_idx, average_weight, exp_factor)
                agent.reset()
            """"""""""""""""""""""""""""""""""""""""""""""""""""""



            if (episode+1)%50 == 0 and rank == 1 and export_flag:
                export_linear_weights(export_path, agent.weights[statespace])



    """""""""""""""""""""""""""
    """"""" MAIN SCRIPT """""""
    """""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""Code Example"""""""""""""""""""""""""""""""""""""""

    n_episode = 4000


    """Initialize agent instance"""
    agent = expSARSA_linear(action_list = [["S1_FRONT", "S1_BACK", "CLOSE_1_1", "OPEN_1_1", "CLOSE_1_2", "OPEN_1_2",
                                            "CLOSE_2_1", "OPEN_2_1", "CLOSE_2_2", "OPEN_2_2",
                                            "S1_LEFT", "S1_RIGHT", "EVAL"],
                                           ["S2_FRONT", "S2_BACK", "CLOSE_1_1", "OPEN_1_1", "CLOSE_1_2", "OPEN_1_2",
                                            "CLOSE_2_1", "OPEN_2_1", "CLOSE_2_2", "OPEN_2_2", "S2_LEFT", "S2_RIGHT",
                                            "S2_ROT_RIGHT", "S2_ROT_LEFT", "S2_UP", "S2_DOWN", "EVAL"]],
                            e_greedy = 0.9
                            )

    run_simulation(agent = agent,
                   n_episodes = n_episode,
                   statespace = 0,
                   convex_option = 0,
                   random_position = 0,
                   export_path =  "/Users/Petros/Documents/RL_Thesis/Python/Fusion/Weights/Phone",
                   export_flag = 0,
                   import_path =  "/Users/Petros/Documents/RL_Thesis/Python/Fusion/Weights/Box",
                   import_flag = 0,
                   comm_frequency = 10)
