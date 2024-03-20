import numpy as np
class ModelHyperparameters:
    """
    This class handles the TP/GP model parameters, this includes the RBF kernel parameters
    (length scale, amplitude) and model parameters (observation noise variance)
    """
    def __init__(self, amplitude, length_scale, observation_noise_variance):
        self._amplitude = amplitude
        self._length_scale = length_scale
        self._observation_noise_variance = observation_noise_variance


    """
    Accessor for amplitude parameter (kernel). Returns float.
    """
    def amplitude(self) -> np.float64:
        return self._amplitude
    """
    Accessor for length scale parameter (kernel). Returns float.
    """
    def length_scale(self) -> np.float64:
        return self._length_scale

    """
    Accessor for observation noise variance parameter (model). Returns float
    """
    def observation_noise_variance(self) -> np.float64:
        return self._observation_noise_variance

    """
    Mutator for amplitude parameter.
    """
    def set_amplitude(self, amplitude_parameter: np.float64):
        self._amplitude = amplitude_parameter


    """
    Mutator for length scale parameter.
    """
    def set_length_scale(self, length_scale_parameter: np.float64):
        self._length_scale = length_scale_parameter

    """
    Mutator for observation noise variance parameter.
    """
    def set_observation_noise_variance(self, observation_noise_variance_parameter):
        self._observation_noise_variance = observation_noise_variance_parameter