import sys
import gym
import numpy as np
from gym_envs.safe_gridworld_env.SafeGridWorld import SafeGridWorld
from ExperimentDataManager import *


class Experiment:
    """
    Setting up the experiment, requires: a Gym environment, the BNP model -,
    either TPs or GPs. Optional: number of episodes and safety threshold d.
    """
    def __init__(self, env: gym.Env, cost_model, reward_model, data_manager: ExperimentDataManager, episodes=100, d=1):
        self.env = env # the safe grid world environment
        self.cost_model = cost_model # the cost model - either a TP or GP
        self.reward_model = reward_model # the reward model - either a TP or GP
        self.experiment_length = episodes # length in episodes of the experiment
        self._uncertainty_threshold = 0.8 # uncertainty threshold for safe decision-making
        self.experiment_data_manager = data_manager # instance of the data manager to write experiment data files
        self.d = d # Task dependant safety tolerance threshold d, d \in CMDP.

    """
    This method starts the experiment, it is called from Main.py
    """
    def start(self):
        cost_predictions = [] # total experiment predictions costs
        cost_observations = [] # total experiment observed costs

        total_observed_cost = 0 # integer used to record total cost observed throughout the experiment
        episode = 0 # episode integer
        task_completion = [] # array to track task completion rate
        env_interaction_steps = 0 # total number of environment interactions

        predicted_rewards = [] # array to track the model's predictive rewards
        observed_rewards = [] # array to track observed rewards.

        episode_rewards = [] # array used to track individual episode rewards
        episode_costs = [] # array used to track individual episode costs
        episode_steps = [] # array used to track individual episode steps (number of steps/decisions/actions required to complete episode/task)

        unsafe_sa_pairs = [] # set of observed unsafe state-action pairs. This is used to compute the safe 'next' action
        num_successful_corrective_samples = 0 # integer used to track the number of successful model updates after observing a new unseen unsafe state.

        while episode < self.experiment_length:
            current_state, info = self.env.reset() # reset environment, observe current state.
            episode_starting_loc = [current_state[0], current_state[1]] # record episode starting coordinates
            episode_goal_loc = [current_state[2], current_state[3]] # record episode goal coordinates
            episode_reward = 0 # episode reward
            done = False # done flag - used to signal if episode has terminated or not
            task_completed = False # task completed flag - used to signal if the task was succesfully completed or not when the episode terminates.

            episode_step = 0 # Integer used to track the number of episode steps.
            episode_cost = 0 # Integer used to track the total cost incurred during an episode
            episode_approx_cost = 0 # Integer used to track the cost approximated by the model
            average_uncertainty_good_action = [] # Array used to track the average uncertainty associated with 'safe' actions
            while not done:
                last_resort_action = False # Flag to check if the backup strategy was employed. Section 4.
                chosen_action = None
                action_costs, action_cost_uncertainties, action_rewards, action_reward_uncertainties = self.sample_models(current_state=current_state)
                chosen_action, last_resort_action = self.compute_safe_action(
                    action_costs=action_costs,
                    action_cost_uncertainties=action_cost_uncertainties,
                    action_rewards=action_rewards,
                    action_reward_uncertainties=action_reward_uncertainties,
                    current_state=current_state,
                    unsafe_sa_pairs=unsafe_sa_pairs
                )

                # Check if the chosen action has resulted in a safety constraint violation.
                # If True, then predict a cost of 1, otherwise predict a cost of 0
                if action_costs[chosen_action] < self.d:
                    cost_predictions.append(1)
                else:
                    cost_predictions.append(0)

                # Record predicted reward, prior to taking the action
                predicted_rewards.append(action_rewards[chosen_action])

                # Perform action, observe reward and cost, transition to next state.
                new_state, reward, done, _, info = self.env.step(chosen_action)

                # Record observed reward
                observed_rewards.append(reward)

                # Increase step by 1
                episode_step += 1

                # Add observed reward to cumulative episode reward.
                episode_reward += reward

                # Check if observed cost violates safety constraint. If True then proceed with online learning. See Section 4.
                if info['state_cost'] * -1 < self.d:
                    # Chosen action was unsafe and resulted in violating the safety constraint.
                    # A cost of c = 1 is added to the agents incurred cost for episode.
                    cost_observations.append(1)
                    episode_cost += 1
                    total_observed_cost += 1
                    print(f"Unsafe State Entered: {new_state}")

                    # If unsafe state is new to the agent, record it.
                    unsafe_sa_pair = [chosen_action, current_state[0], current_state[1], current_state[5], current_state[6]]
                    if unsafe_sa_pair not in unsafe_sa_pairs:
                        unsafe_sa_pairs.append(unsafe_sa_pair)

                    # This block checks if the update conditions are met, if True the model is updated with the new observed cost
                    # and retrained.
                    action_costs, action_cost_uncertainties, _, _ = self.sample_models(current_state=current_state)
                    print(f"Cost of unsafe action before updating: {action_costs[chosen_action]}, uncertainty associated: {action_cost_uncertainties[chosen_action]}")
                    if action_costs[chosen_action] >= self.d or action_cost_uncertainties[chosen_action] == np.min(action_cost_uncertainties) and last_resort_action == False:
                        print("Amending observation and retraining")
                        cost_observation = [float(chosen_action), current_state[0], current_state[1], current_state[5], current_state[6]]
                        self.cost_model.update_observation_in_buffers(x_observation=cost_observation, new_y=(info['state_cost'] * -1))
                        # After Check
                        r_action_costs, r_action_cost_uncertainties, _, _ = self.sample_models(current_state=current_state)
                        print(f"Cost of unsafe action after updating: {r_action_costs[chosen_action]}, uncertainty associated: {r_action_cost_uncertainties[chosen_action]}")

                        if r_action_costs[chosen_action] < self.d:
                            num_successful_corrective_samples += 1


                else:
                    cost_observations.append(0)
                    average_uncertainty_good_action.append(action_cost_uncertainties[chosen_action])

                task_completed = info['task_completed']
                if done == True:
                    task_completion.append(info['task_completed'])

                current_state = new_state

            # Penalise cost rate by disregarding episode steps if task not completed or if any cost is incurred during the episode.
            if task_completed == True and episode_cost == 0:
                env_interaction_steps += episode_step

            # Episode End
            # Data recording and metric computations
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            # Actual cost rate incurred by SRL agent. See paper.
            observed_cost_rate = round(total_observed_cost / env_interaction_steps, 3) if total_observed_cost > 0 and env_interaction_steps > 0 else 0

            # Total error between models a priori approximation of immediate future cost and the observed cost once an action is taken.
            total_cost_error = round(np.abs(np.mean((np.array(cost_observations) - np.array(cost_predictions)))), 3) if len(cost_predictions) == len(cost_observations) else "N/A"

            # The mean squared error (MSE) of the reward approximation
            reward_approximation_mse = round(np.mean((np.array(observed_rewards) - np.array(predicted_rewards)) ** 2), 3) if len(observed_rewards) == len(predicted_rewards) else "N/A"

            # Task success rate. Number of tasks successfully complete / total number of tasks
            task_success_rate = round((np.mean(task_completion) * 100), 3)
            episode_steps.append(episode_step)
            print(
                f"Episode: {episode + 1}, "
                f"Mean Reward: {round(np.mean(episode_rewards), 3)}, "  # Mean reward acheived by SRL agent
                f"Mean Cost: {round(np.mean(episode_costs), 3)}, "  # Mean cost incurred by SRL agent. AKA Average number of constraint violations.
                f"Reward Approximation MSE: {reward_approximation_mse}, "  # See above
                f"Observed cost: {episode_cost}, "  # Observed (actual) cost incurred by SRL agent
                f"Approximated cost: {episode_approx_cost}, "  # Approximated (perceived) cost incurred by SRL agent
                f"Observed Cost Rate: {observed_cost_rate}, "  # See above.
                f"Total Cost: {total_observed_cost}, "  # Total cost incurred over all episodes. 
                f"Absolute Cost Prediction Error: {total_cost_error}, "
                f"Task completed: {task_completed}, "  # Was the task completed.
                f"Number of tasks completed: {np.sum(task_completion)}, "  # Number of tasks completed by SRL agent.
                f"Number of steps in episode: {episode_step}, "  # Number of steps (actions) taken by SRL agent during episode.
                f"Episode starting location: {episode_starting_loc}, "  # The starting location (coordinates) of the SRL agent. 
                f"Episode goal location: {episode_goal_loc}, "  # The goal location (coordinates) for the SRL agent to try and reach safely.
                f"Task success proportion: {task_success_rate}%, "  # See above
                f"Mean Cost Uncertainty Good actions: {np.mean(average_uncertainty_good_action)}, " # The mean variance associated with 'safe' and conducive actions.
                f"Mean Episode Steps: {np.mean(episode_steps)}, " # Mean steps to complete episode
                f"Number of Successful Corrective Model Updates: {num_successful_corrective_samples}" # Number of attempts to correct model knowledge that were successful.
            )

            # Data recording
            ats = lambda a : "(" + str(int(a[0])) + " " + str(int(a[1])) + ")"

            episode_record = [
                episode + 1,
                round(np.mean(episode_rewards), 3),
                round(np.mean(episode_costs), 3),
                reward_approximation_mse,
                episode_cost,
                episode_approx_cost,
                observed_cost_rate,
                total_observed_cost,
                total_cost_error,
                task_completed,
                np.sum(task_completion),
                episode_step,
                ats(episode_starting_loc),
                ats(episode_goal_loc),
                task_success_rate,
                np.mean(average_uncertainty_good_action),
                np.mean(episode_steps),
                num_successful_corrective_samples
            ]
            for i in range(len(episode_record)):
                episode_record[i] = str(episode_record[i])
            data_entry = ",".join(episode_record)
            self.experiment_data_manager.append_data_entry(data_entry)



            episode += 1


        # Return the average reward achieved by the SRL agent and the average cost incurred by the SRL agent.
        return round(np.mean(episode_rewards), 3), round(np.mean(episode_costs), 3)


    """
    This method samples the cost and reward models for a mean prediction and associated uncertainty. 
    This is done for each action a \in A.
    """
    def sample_models(self, current_state):

        action_costs = []
        action_cost_uncertainties = []
        action_rewards = []
        action_reward_uncertainties = []

        for action in range(self.env.action_space.n):
            cost_model_observation = [float(action), current_state[0], current_state[1], current_state[5], current_state[6]]
            action_cost, cost_uncertainty = self.cost_model.sample_posterior_predictive(x_observation=cost_model_observation)
            action_costs.append(action_cost)
            action_cost_uncertainties.append(cost_uncertainty)

            reward_model_observation = [float(action), current_state[0], current_state[1], current_state[2], current_state[3]]
            action_reward, reward_uncertainty = self.reward_model.sample_posterior_predictive(x_observation=reward_model_observation)
            action_rewards.append(action_reward)
            action_reward_uncertainties.append(reward_uncertainty)

        return action_costs, action_cost_uncertainties, action_rewards, action_reward_uncertainties


    """
    This method implements part of Algorithm  (1) in the paper. This method attempts to compute a safe and conducive action
    based on action costs, action cost uncertainty action rewards, current state and known unsafe state-action pairs.
    """
    def compute_safe_action(self, action_costs, action_cost_uncertainties, action_rewards, action_reward_uncertainties, current_state, unsafe_sa_pairs):

        last_resort_action = False
        highest_reward_action = None
        highest_reward = -np.inf


        modified_action_costs = action_costs[:]
        modified_action_cost_uncertainties = action_cost_uncertainties[:]
        modified_action_rewards = action_rewards[:]

        for i in range(self.env.action_space.n):

            observation = [i, current_state[0], current_state[1], current_state[5], current_state[6]]
            if observation in unsafe_sa_pairs:
                modified_action_costs[i] = -np.inf
                modified_action_cost_uncertainties[i] = np.inf
                modified_action_rewards[i] = -np.inf



        for action, (cost, cost_uncertainty, reward) in enumerate(zip(modified_action_costs, modified_action_cost_uncertainties, modified_action_rewards)):
            if cost_uncertainty <= self._uncertainty_threshold and cost >= self.d:
                # Safe Action
                if reward > highest_reward:
                    highest_reward_action = action
                    highest_reward = reward

        if highest_reward_action == None:
            last_resort_action = True
            highest_reward_action = np.argmax(modified_action_rewards)

        return highest_reward_action, last_resort_action



