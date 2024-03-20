import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow_probability.python.distributions
from ModelHyperparameters import ModelHyperparameters

class StudentTProcess:

    def __init__(self, duplicate_observation_limit):
        # TPs Kernel Parameter Initilisation
        self._kernel_parameters = ModelHyperparameters(1.0, 1.0, 1.0)
        # TPs degrees of freedom (DOF) parameter nu ($\nu$) - nu describes the tail weight of the underlying Student's T distribution
        self.nu = 3
        # TPs X, Y Observation Buffers - Initialised to empty arrays
        self._x_observation_buffer = []
        self._y_observation_buffer = []

        # TPs prior samples - initialied to empty arrays
        self._fixed_x_prior = []
        self._fixed_y_prior = []

        # Sample control
        # TPs current memory size - number of samples currently in use. Initialised to zero
        self._memory_size = 0
        self._max_memory_size = 999999 # Max capacity
        self._duplicate_observation_limit = duplicate_observation_limit

        # Global state flags
        # Is GP trained - Initialised to False
        self._is_trained = False
        self._adam_learning_rate = 0.1

        # Model inference flags
        # Number of function realizations to draw from posterior predictive distribution og GP - p(y | f, X)
        self._num_function_samples = 150
        # The proportion of the memory buffer to be used in the prior distribution when inferring using the GP
        self._prior_proportion = 0.347
        self.model = "TP"

    """
    This method is an accessor for the TP model's current memory size.
    The self._memory_size variable denotes the current number of (x,y) observations
    available to the model.
    """
    def memory_size(self) -> int:
        return self._memory_size

    """
    This method is an accessor for the TP model's _is_trained flag. This flag is True if
    at least one round of training has taken place, i.e. can the model be used to exploit 
    current knowledge of the CMDP.
    """
    def is_model_trained(self) -> bool:
        return self._is_trained

    """
    This method is an accessor for the (x,y) observation buffers, it returns the current contents
    of the observation buffers as lists.
    """
    def fetch_buffer_contents(self):
        return self._x_observation_buffer[:], self._y_observation_buffer[:]

    """
    This method is a mutator for the x,y observation buffers, so that outliers can be injected
    into the training data.
    """
    def set_buffers(self, x_buffer, y_buffer):
        self._x_observation_buffer = x_buffer
        self._y_observation_buffer = y_buffer

    """
    This is an internal class method used to count the number of duplicate occurrences of a
    given x observation in the buffers.
    """
    def _count_duplicate_samples(self, sample):

        duplicates = 0

        for i in range(len(self._x_observation_buffer)):
            if sample == self._x_observation_buffer[i]:
                duplicates += 1

        return duplicates

    """
    This method adds a new x,y observation pair to the GP model's buffers.
    """
    def add_sample(self, x_observation, y_observation):

        if self._count_duplicate_samples(x_observation) < self._duplicate_observation_limit:

            if self._memory_size == self._max_memory_size:

                del self._x_observation_buffer[self._memory_size - 1]
                del self._y_observation_buffer[self._memory_size - 1]
                self._memory_size -= 1

            self._x_observation_buffer.insert(0, x_observation)
            self._y_observation_buffer.insert(0, y_observation)
            self._memory_size += 1

    """
    This method colloquially known as 'train' optimizes the TPs kernel hyperparameters to find the best
    fit for the regression model.
    """
    def train(self, training_epochs):

        def build_conditional_distribution(amplitude, length_scale, observation_noise_variance):
            kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
            tp = tfp.distributions.StudentTProcess(
                df=self.nu,
                kernel=kernel,
                index_points=self._x_observation_buffer,
                observation_noise_variance=observation_noise_variance
            )

            return tp

        tp_joint_model = tensorflow_probability.distributions.JointDistributionNamed(
            dict(
                amplitude=tensorflow_probability.distributions.LogNormal(loc=0.0, scale=np.float64(1.0)),
                length_scale=tensorflow_probability.distributions.LogNormal(loc=0.0, scale=np.float64(1.0)),
                observation_noise_variance=tensorflow_probability.distributions.LogNormal(loc=0.0,scale=np.float64(1.0)),
                observations=build_conditional_distribution
            )
        )

        # This bijector constrains the models hyperparameters to remain positive, regardless of the transformation applied to them during training.
        positive_bijector = tensorflow_probability.bijectors.Shift(np.finfo(np.float64).tiny)(tensorflow_probability.bijectors.Exp())

        # Trainable realizations of the kernel's & model's hyperparameters
        amplitude_var = tfp.util.TransformedVariable(initial_value=1., bijector=positive_bijector, name='amplitude', dtype=np.float64)
        length_scale_var = tfp.util.TransformedVariable(initial_value=1.,bijector=positive_bijector,name='length_scale', dtype=np.float64)
        observation_noise_variance_var = tfp.util.TransformedVariable(initial_value=1.,bijector=positive_bijector, name='observation_noise_variance', dtype=np.float64)

        trainable_variables = [v.trainable_variables[0] for v in [amplitude_var, length_scale_var, observation_noise_variance_var]]

        def target_log_prob(amplitude, length_scale, observation_noise_variance):

            return tp_joint_model.log_prob({
                'amplitude': amplitude,
                'length_scale': length_scale,
                'observation_noise_variance': observation_noise_variance,
                'observations': self._y_observation_buffer
            })

        optimizer = tf.optimizers.legacy.Adam(learning_rate=self._adam_learning_rate)
        @tf.function(autograph=False, jit_compile=False, reduce_retracing=True)
        def train_model():

            with tf.GradientTape() as tape:
                loss = -target_log_prob(amplitude=amplitude_var, length_scale=length_scale_var,
                                        observation_noise_variance=observation_noise_variance_var)

            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            return loss

        # print("Training Student-t Process with SE Kernel using Adam optimization...")
        for epoch in range(training_epochs):
            loss = train_model()
            # if epoch % 10 == 0:
                # print(f"Training epoch: {epoch}, loss: {loss}")

        # print("TP Training Complete")
        self._is_trained = True
        self._kernel_parameters.set_amplitude(amplitude_parameter=amplitude_var.numpy())
        self._kernel_parameters.set_length_scale(length_scale_parameter=length_scale_var.numpy())
        self._kernel_parameters.set_observation_noise_variance(observation_noise_variance_parameter=observation_noise_variance_var.numpy())

        # print(f"Parameters:\nAmplitude: {amplitude_var.numpy()}\nLength scale: {length_scale_var.numpy()}, obnv: {observation_noise_variance_var.numpy()}")

    """
    This method samples function evaluations from the TPs posterior predictive distribution. It is used by the agent
    to make predictions about the reward/cost given the (s,a) pair
    """
    def sample_posterior_predictive(self, x_observation) -> (np.float64, np.float64):
        prior_x, prior_y = self._generate_stochastic_prior(x_observation=x_observation)
        x_observation = np.asarray([x_observation], dtype=np.float64)

        posterior_predictive = tfp.distributions.StudentTProcessRegressionModel(
            df=self.nu,
            kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(self._kernel_parameters.amplitude(), self._kernel_parameters.length_scale()),
            index_points=x_observation,
            observation_index_points=prior_x,
            observations=prior_y,
            observation_noise_variance=self._kernel_parameters.observation_noise_variance()
        )

        function_samples = posterior_predictive.sample(self._num_function_samples)
        mean_sample = tf.reduce_mean(function_samples).numpy()
        max_sample = tf.reduce_max(function_samples).numpy()
        uncertainty = tf.math.reduce_variance(function_samples).numpy()

        return mean_sample, uncertainty

    """
    This method generates a 'stochastic' prior. A stochastic prior in this context refers to the information
    the posterior predictive distribution will be conditioned on. We generate n random samples from seen data
    and then attempt to add up to 10 samples with matching x values to ensure the model has adequate inference 
    ability w.r.t the specific x observation if it has seen a similar one in the past.
    """
    def _generate_stochastic_prior(self, x_observation) -> (list, list):

        prior_samples_size = round(self._memory_size * self._prior_proportion) if round(self._memory_size * self._prior_proportion) > 400 else 400
        prior_samples_size = 400
        prior_x = []
        prior_y = []

        if len(self._fixed_x_prior) > 0:
            prior_x = self._fixed_x_prior[-200:]
            prior_y = self._fixed_y_prior[-200:]

        observation_indices = np.random.choice(self._memory_size, prior_samples_size - len(prior_x), replace=False)
        # observation_indices = np.random.choice(self._memory_size, prior_samples_size, replace=False)
        for i in range(len(observation_indices)):
            prior_x.append(self._x_observation_buffer[observation_indices[i]])
            prior_y.append(self._y_observation_buffer[observation_indices[i]])

        temp_x_buffer = self._x_observation_buffer[:]
        temp_y_buffer = self._y_observation_buffer[:]

        temp_x_buffer.extend(self._fixed_x_prior)
        temp_y_buffer.extend(self._fixed_y_prior)

        samples_added = 0
        max_samples = 10
        for i in range(max_samples):
            try:
                index = temp_x_buffer.index(x_observation)
                prior_x.append(temp_x_buffer[index])
                prior_y.append(temp_y_buffer[index])
                del temp_x_buffer[index]
                del temp_y_buffer[index]
                samples_added += 1
            except ValueError:
                break

        return prior_x, prior_y

    """
    This method is used to update the model's memory buffer during online learning. The 
    formal algorithm using this functionality is given as Algorithm (2) in the paper.
    """
    def update_observation_in_buffers(self, x_observation, new_y):
        num_updates_attempted = 0
        num_updates_made = 0
        for i in range(len(self._x_observation_buffer)):

            if self._x_observation_buffer[i] == x_observation:
                num_updates_attempted += 1
                if self._y_observation_buffer[i] != new_y:
                    self._y_observation_buffer[i] = new_y
                    num_updates_made += 1

        print(f"{num_updates_attempted} observations identified, {num_updates_made} updates made")
        if num_updates_made > 0:
            self.train(training_epochs=50)
        else:
            if num_updates_attempted == 0:
                for i in range(self._duplicate_observation_limit):
                    self.add_sample(x_observation=x_observation, y_observation=new_y)
                print("No data found on state-action, adding sample and retraining")
                self.train(training_epochs=50)

