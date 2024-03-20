import sys
from gym_envs.safe_gridworld_env.SafeGridWorld import SafeGridWorld
from ExperimentDataManager import *
from GaussianProcess import *
from StudentTProcess import *
from Experiment import *
"""
    This method uses a stochastic control policy to collect n environment observations over m episodes. 
    The method will either collect state-action-cost observations or state-action-reward observations, depending
    on the 'function' flag i.e., function = "c" is cost, and function = "r" is reward.
"""
def stochastically_sample_function(env: SafeGridWorld, function: str, episodes=100):
    assert function == "c" or function == "r"
    model_x_data = []
    model_y_data = []

    for episode in range(episodes):
        current_state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # The .sample() method uniformly randomly samples the action space A, the probability of any given action is therefore: p(a) = \frac{1}{|A|}
            action = env.action_space.sample()

            # state-action-cost observation
            if function == "c":
                x_observation = [action, current_state[0], current_state[1], current_state[5], current_state[6]]
            # state-action-reward observation
            elif function == "r":
                x_observation = [action, current_state[0], current_state[1], current_state[2], current_state[3]]

            model_x_data.append(x_observation)

            # Step using chosen action
            new_state, reward, done, _, info = env.step(action)

            # See above
            if function == "c":
                y_observation = info['state_cost'] * -1
            elif function == "r":
                y_observation = reward

            model_y_data.append(y_observation)

            episode_reward += reward
            current_state = new_state

    return model_x_data, model_y_data

# Parsing experiment parameters
try:
    assert len(sys.argv) == 5
except AssertionError:
    print("Expecting\n$ python main.py [experiment length] [outlier magnitude] [outlier ratio] [number of switch states]")
    exit(0)

# Preliminaries
goal_location = [7,7] # Example goal location. This pertains to (x,y) coordinates on a 2d gridworld
cost_model_duplicate_observation_limit = 12 # duplicate observation limit for cost model i.e. no more than 12 matching (x,y) observations
reward_model_duplicate_observation_limit = 1 # duplicate observation limit for reward model i.e. no more than 1 matching (x,y) observations
training_epochs = 50 # Number of training iterations for the adaptive moment estimation (ADAM) algorithm
safety_tolerance = 1 # Safety tolerance. d = 1 \in CMDP
num_switch_states = 10 # Number of unsafe states that will change location under covariate shift. This serves as the degree of covariate shift
num_unsafe_states = 15 # Total number of unsafe states in the environment

experiment_length = int(sys.argv[1]) # Experiment length in episodes. All experiments reported in the paper ran for 250 episodes.
outlier_magnitude = float(sys.argv[2]) # Outlier magnitude
outlier_ratio = float(sys.argv[3]) # Outlier ration - the proportion of training data that are outliers.
num_switch_states = int(sys.argv[4]) # See above

# This is the instance of our SafeGridWorld Gym Environment.
# The fixed_starting_loc flag tells the enviroment to randomise the agent's starting location at the genesis of each episode.
# The fixed_goal_loc flag tells the environment to randomise the agent's goal location at the genesis of each episode.
env = SafeGridWorld(goal_location=goal_location, fixed_starting_loc=False, fixed_goal_loc=False, num_unsafe_states=num_unsafe_states, num_switch_states=num_switch_states)

# The instantiation of each cost and reward model to be used throughout the experiment
tp_cost_model = StudentTProcess(duplicate_observation_limit=cost_model_duplicate_observation_limit)
tp_reward_model = StudentTProcess(duplicate_observation_limit=reward_model_duplicate_observation_limit)
gp_cost_model = GaussianProcess(duplicate_observation_limit=cost_model_duplicate_observation_limit)
gp_reward_model = GaussianProcess(duplicate_observation_limit=reward_model_duplicate_observation_limit)

# outlier flags
outliers_in_cost_model = True
outliers_in_reward_model = True

# Sample c() and r() to generate the offline datasets
cost_x_data, cost_y_data = stochastically_sample_function(env=env, function="c", episodes=100)
reward_x_data, reward_y_data = stochastically_sample_function(env=env, function="r", episodes=100)


# Populate the cost model's observation buffer with data from the offline datasets
for i in range(len(cost_x_data)):
    action = int(cost_x_data[i][0])
    x_observation = list(cost_x_data[i][1:])
    y_observation = float(cost_y_data[i])

    tp_cost_model.add_sample(x_observation=list(cost_x_data[i]), y_observation=y_observation)
    gp_cost_model.add_sample(x_observation=list(cost_x_data[i]), y_observation=y_observation)
# Populate the reward model's observation buffer with data from offline datasets
for i in range(len(reward_x_data)):
    action = int(reward_x_data[i][0])
    x_observation = list(reward_x_data[i][1:])
    y_observation = float(reward_y_data[i])

    tp_reward_model.add_sample(x_observation=list(reward_x_data[i]), y_observation=y_observation)
    gp_reward_model.add_sample(x_observation=list(reward_x_data[i]), y_observation=y_observation)


# Find the buffer sizes of each model and determine the smallest
buffer_sizes = [tp_cost_model.memory_size(), tp_reward_model.memory_size(), gp_cost_model.memory_size(), gp_reward_model.memory_size()]
min_buffer_size = np.min(buffer_sizes)
# Choose n uniformly randomly distributed indices to be outliers
outlier_indices = np.random.choice(min_buffer_size, round(outlier_ratio * min_buffer_size), replace=False)

# Populate the observation buffers with outliers
if outliers_in_reward_model == True:
    x_buffer, y_buffer = tp_reward_model.fetch_buffer_contents()
    x_buffer, y_buffer = np.array(x_buffer), np.array(y_buffer)

    # See section 5
    x_buffer[outlier_indices] *= outlier_magnitude
    y_buffer[outlier_indices] *= outlier_magnitude
    tp_reward_model.set_buffers(x_buffer=x_buffer.tolist(), y_buffer=list(y_buffer))

    x_buffer, y_buffer = gp_reward_model.fetch_buffer_contents()
    x_buffer, y_buffer = np.array(x_buffer), np.array(y_buffer)

    x_buffer[outlier_indices] *= outlier_magnitude
    y_buffer[outlier_indices] *= outlier_magnitude
    gp_reward_model.set_buffers(x_buffer=x_buffer.tolist(), y_buffer=list(y_buffer))

# Populate the observation buffers with outliers
if outliers_in_cost_model == True:
    x_buffer, y_buffer = tp_cost_model.fetch_buffer_contents()
    x_buffer, y_buffer = np.array(x_buffer), np.array(y_buffer)

    # See section 5
    x_buffer[outlier_indices] *= outlier_magnitude
    y_buffer[outlier_indices] *= outlier_magnitude
    tp_cost_model.set_buffers(x_buffer=x_buffer.tolist(), y_buffer=list(y_buffer))

    x_buffer, y_buffer = gp_cost_model.fetch_buffer_contents()
    x_buffer, y_buffer = np.array(x_buffer), np.array(y_buffer)

    x_buffer[outlier_indices] *= outlier_magnitude
    y_buffer[outlier_indices] *= outlier_magnitude
    gp_cost_model.set_buffers(x_buffer=x_buffer.tolist(), y_buffer=list(y_buffer))


# Train each model with offline data
gp_reward_model.train(training_epochs=training_epochs)
gp_cost_model.train(training_epochs=training_epochs)
tp_reward_model.train(training_epochs=training_epochs)
tp_cost_model.train(training_epochs=training_epochs)

# Print experiment metadata
print("Training Completed...")
print(
    f"Outlier information:\n"
    f"Outliers in Cost Model: {outliers_in_cost_model}\n"
    f"Outliers in Reward Model: {outliers_in_reward_model}\n"
    f"Outlier Proportion: {outlier_ratio}\n"
    f"Outlier Magnitude: {outlier_magnitude}\n"
    f"Experiment Length: {experiment_length}\n"
    f"Number of Unsafe States: {num_unsafe_states}\n"
    f"Number of Unsafe Switch States: {num_switch_states}\n"
)
# Perform covariate shift
env.shift_distribution()

# Setup data recording facilities
directory = "ExperimentRuns"
gp_experiment_id = f"experiment_gp_om={outlier_magnitude}_op={outlier_ratio}_epi={experiment_length}_nss={num_switch_states}.csv"
tp_experiment_id = f"experiment_tp_om={outlier_magnitude}_op={outlier_ratio}_epi={experiment_length}_nss={num_switch_states}.csv"
# Data we will record
data_fields = [
    "episode",
    "mean_reward",
    "mean_cost",
    "reward_approximation_mse",
    "observed_cost",
    "approximated_cost",
    "observed_cost_rate",
    "total_cost",
    "absolute_cost_prediction_error",
    "task_completed",
    "number_of_tasks_completed",
    "steps",
    "starting_location",
    "goal_location",
    "task_success_rate",
    "safe_actions_mean_uncertainty",
    "mean_episode_steps",
    "number_successful_corrective_updates"
]
data_header = ",".join(data_fields)
# Begin TP experiment.
tp_experiment_data_manager = ExperimentDataManager(directory=directory, filename=tp_experiment_id, data_header=data_header)
print("Running TP Simulation...")
tp_experiment = Experiment(env=env, cost_model=tp_cost_model, reward_model=tp_reward_model, episodes=experiment_length, d=safety_tolerance, data_manager=tp_experiment_data_manager)
tp_mean_reward, tp_mean_cost = tp_experiment.start()
print("--------- TP Experiment END --------\n\n\n")

# Begin GP experiment
gp_experiment_data_manager = ExperimentDataManager(directory=directory, filename=gp_experiment_id, data_header=data_header)
print("Running GP Simulation...")
gp_experiment = Experiment(env=env, cost_model=gp_cost_model, reward_model=gp_reward_model, episodes=experiment_length, d=safety_tolerance, data_manager=gp_experiment_data_manager)
gp_mean_reward, gp_mean_cost = gp_experiment.start()
print("--------- GP Experiment END --------\n\n\n")


cost_result = "TP is the safer model" if tp_mean_cost <= gp_mean_cost else "GP is the safer model"
reward_result = "TP better maximises the reward function" if tp_mean_reward >= gp_mean_reward else "GP better maximises the reward function"
print(f"\n\n\nExperiment Results:\nCost Result: {cost_result}\nReward Result: {reward_result}\n\n\n")



