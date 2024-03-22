import gym
from gym import spaces
import numpy as np
import pygame

class SafeGridWorld(gym.Env):

    def __init__(self, goal_location = [9, 9], fixed_starting_loc=False, fixed_goal_loc=False, num_unsafe_states=10, num_switch_states=1):
        super(SafeGridWorld, self).__init__()
        self._bias_starting_state = False
        self._add_unsafe_states = False
        self._bias_starting_range_ub = None
        self._bias_starting_range_lb = None
        self._distribution_shift_factor = 2
        self._fixed_starting_loc = fixed_starting_loc
        self._fixed_goal_loc = fixed_goal_loc
        if self._fixed_goal_loc:
            self._set_goal_loc = goal_location

        self._num_unsafe_states = num_unsafe_states
        self._additional_unsafe_states = num_switch_states
        self._world_size = 10
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        self.action_to_str = {
            0: "RIGHT",
            1: "DOWN",
            2: "LEFT",
            3: "UP",
        }

        self.action_str_to_action = {
            "RIGHT": 0,
            "DOWN": 1,
            "LEFT": 2,
            "UP": 3,
        }

        self._unsafe_states = []
        self._agent_start_location = []

        self._agent_start_location, self._unsafe_states, self._agent_goal_location = self._generate_dynamic_world_objects()

        self._agent_current_location = self._agent_start_location

        self._agent_step = 0
        self._agent_max_step = (self._world_size*self._world_size)*1

        self._default_cost = 1
        self.MAX_REWARD = 50

        self.observation_space = spaces.Box(0, self._world_size-1, shape=(5,), dtype=np.float64)

        self._AGENT_START_ = 1
        self._AGENT_CURRENT_ = 2
        self._AGENT_GOAL_ = 50
        self._UNSAFE_STATE_ = -1

        self.render_mode = None
        self.test_fidelity()


    def shift_distribution(self):

        print("Unsafe states before shift")
        print(self._unsafe_states)

        self._bias_starting_range_ub = self._world_size*self._distribution_shift_factor
        self._bias_starting_range_lb = self._bias_starting_range_ub-self._world_size

        # self._world_size *= self._distribution_shift_factor
        # self.observation_space = spaces.Box(0, self._world_size-1, shape=(5,), dtype=np.float64)
        self._agent_max_step = (self._world_size * self._world_size) * 1
        # self._bias_starting_state = True
        self._add_unsafe_states = True
        self._agent_start_location, self._unsafe_states, self._agent_goal_location = self._generate_dynamic_world_objects()
        # self._num_unsafe_states += 5
        print("Unsafe States after shift")
        print(self._unsafe_states)

    def test_fidelity(self):
        assert list(self._agent_start_location) != list(self._agent_goal_location)
        assert list(self._agent_start_location) not in self._unsafe_states
        assert list(self._agent_goal_location) not in self._unsafe_states



    def _generate_dynamic_world_objects(self):
        goal_state = [np.random.randint(0, self._world_size-1), np.random.randint(0, self._world_size-1)] if self._fixed_goal_loc == False else self._set_goal_loc

        while (goal_state in self._unsafe_states):
            goal_state = [np.random.randint(0, self._world_size-1), np.random.randint(0, self._world_size-1)] if self._fixed_goal_loc == False else self._set_goal_loc

        if not self._bias_starting_state:
            agent_start_ub = self._world_size - 1
            agent_start_lb = 0
        elif self._bias_starting_state:
            agent_start_ub = 11
            agent_start_lb = 10

        agent_start = [np.random.randint(agent_start_lb, agent_start_ub), np.random.randint(agent_start_lb, agent_start_ub)] if self._fixed_starting_loc == False else [0, 0]

        while (agent_start in self._unsafe_states) or agent_start == list(goal_state):
            agent_start = [np.random.randint(agent_start_lb, agent_start_ub), np.random.randint(agent_start_lb, agent_start_ub)] if self._fixed_starting_loc == False else [0, 0]

        world_objects = [
            agent_start,
            goal_state,
            self._surrounding_states(goal_state)[0]
        ]
        if self._fixed_starting_loc == False:
            world_objects.append([0,0])

        if len(self._unsafe_states) == self._num_unsafe_states and self._add_unsafe_states == False:
             return np.array(world_objects[0], dtype=np.float64), self._unsafe_states, np.array(goal_state, dtype=np.float64)
        elif self._add_unsafe_states == True:
            print(f"Moving {self._additional_unsafe_states} unsafe states")
            self._add_unsafe_states = False

            old_unsafe_states_indices = np.random.choice(self._num_unsafe_states, (self._num_unsafe_states-self._additional_unsafe_states), replace=False)
            new_unsafe_states = []
            for i in range(len(old_unsafe_states_indices)):
                new_unsafe_states.append(self._unsafe_states[old_unsafe_states_indices[i]])
                world_objects.append(self._unsafe_states[old_unsafe_states_indices[i]])
                world_objects.extend(self._surrounding_states(self._unsafe_states[old_unsafe_states_indices[i]]))

            world_objects.extend(new_unsafe_states)
            n_us = []
            while len(new_unsafe_states) < self._num_unsafe_states:
                unsafe_state = [np.random.randint(0, self._world_size - 1), np.random.randint(0, self._world_size - 1)]

                if unsafe_state not in world_objects:
                    new_unsafe_states.append(unsafe_state)
                    world_objects.append(unsafe_state)
                    world_objects.extend(self._surrounding_states(unsafe_state))
                    n_us.append(unsafe_state)

            print(f"New unsafe states: {n_us}")
            return np.array(world_objects[0], dtype=np.float64), new_unsafe_states, np.array(goal_state, dtype=np.float64)

        unsafe_states = []

        while len(unsafe_states) < self._num_unsafe_states:
            unsafe_state = [np.random.randint(0, self._world_size-1), np.random.randint(0, self._world_size-1)]

            if unsafe_state not in world_objects:
                unsafe_states.append(unsafe_state)
                world_objects.append(unsafe_state)
                world_objects.extend(self._surrounding_states(unsafe_state))

        return np.array(world_objects[0], dtype=np.float64), unsafe_states, np.array(goal_state, dtype=np.float64)


    def _surrounding_states(self, state):
        surrounding_states = []
        for i in range(self.action_space.n):
            direction = self._action_to_direction[i]
            surrounding_state = np.clip(np.array(state) + direction, 0, self._world_size - 1)
            if list(surrounding_state) != state:
                surrounding_states.append(list(surrounding_state))
        return surrounding_states

    def _get_obs(self):
        # Observation contains agent's current (x,y), the goal's (x,y)-
        observation = np.concatenate((
            self._agent_current_location,
            self._agent_goal_location,
        ))

        observation = np.concatenate((observation, [self._unnormalised_l1_norm(self._agent_current_location, self._agent_goal_location)]))
        distance_to_unsafe_state, unsafe_state_location = self._distance_to_closest_unsafe_state()

        observation = np.concatenate((observation, unsafe_state_location))
        observation = np.concatenate((observation, [distance_to_unsafe_state]))
        return observation
    def _get_info(self):
        return {"task_completed": self._task_completed(), "state_cost": self._get_state_cost()}

    def _distance_to_closest_unsafe_state(self):

        distances = []

        for i in range(len(self._unsafe_states)):
            is_in_state = np.array_equal(self._agent_current_location, np.array(self._unsafe_states[i]))
            if is_in_state == True:
                distances.append(0)
            else:
                distances.append(self._unnormalised_l2_norm(np.array(self._unsafe_states[i]), self._agent_current_location))

        return np.min(distances), self._unsafe_states[np.argmin(distances)]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_start_location, self._unsafe_states, self._agent_goal_location = self._generate_dynamic_world_objects()
        self._agent_current_location = self._agent_start_location
        self._agent_step = 0
        observation = self._get_obs()
        info = self._get_info()
        self.test_fidelity()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_current_location = np.clip(self._agent_current_location + direction, 0, self._world_size-1)
        terminated = self._has_terminated()
        reward = self._reward_functon()
        observation = self._get_obs()
        info = self._get_info()
        self._agent_step += 1
        return observation, np.float64(reward), terminated, False, info

    def _has_terminated(self):
        # Termination condition 1
        task_completed = self._task_completed()
        # Termination condition 2
        exceeded_steps = self._agent_step == self._agent_max_step
        return task_completed or exceeded_steps

    def _task_completed(self):
        return np.array_equal(self._agent_current_location, self._agent_goal_location)

    def _reward_functon(self):
        return -(self._unnormalised_l2_norm(l1=self._agent_current_location, l2=self._agent_goal_location))

    def _unnormalised_l1_norm(self, l1, l2):
        return np.linalg.norm(l1 - l2, ord=1)

    def _unnormalised_l2_norm(self, l1, l2):
        return np.linalg.norm((l1 - l2), ord=2)

    def _normalised_l2_norm(self, l1, l2):
        l2_norm = np.linalg.norm((l1 - l2), ord=2)
        return l2_norm / (1.0 + l2_norm)

    def _get_state_cost(self):
       distance_to_unsafe_state, unsafe_state = self._distance_to_closest_unsafe_state()
       return -(distance_to_unsafe_state)


    def set_fixed_starting(self, value):
        self._fixed_starting_loc = value

    def print(self):
        world = np.zeros((self._world_size-1, self._world_size-1))
        world[int(self._agent_start_location[0])][int(self._agent_start_location[1])] = self._AGENT_START_
        world[int(self._agent_current_location[0])][int(self._agent_current_location[1])] = self._AGENT_CURRENT_ if list(self._agent_current_location) != list(self._agent_start_location) else self._AGENT_START_
        world[int(self._agent_goal_location[0])][int(self._agent_goal_location[1])] = self._AGENT_GOAL_
        for i in range(self._num_unsafe_states):
            world[self._unsafe_states[i][0]][self._unsafe_states[i][1]] = self._UNSAFE_STATE_

        print("\n")
        print(world)
        print("\n")
