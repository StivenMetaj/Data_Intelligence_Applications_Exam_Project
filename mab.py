import numpy as np


class Learner():
    def __init__(self, n_arms, arms):
        self.n_arms = n_arms
        self.arms = arms
        self.t = 0
        self.rewards = []
        self.arm_pulled = []

    def update_observation(self, pulled_arm, reward):
        self.rewards.append(reward)
        self.arm_pulled.append(pulled_arm)


class TS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms=n_arms, arms=arms)
        self.beta_param = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.array(self.arms) * np.random.beta(self.beta_param[:, 0], self.beta_param[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.update_observation(pulled_arm, reward)
        self.beta_param[pulled_arm, 0] = self.beta_param[pulled_arm, 0] + reward
        self.beta_param[pulled_arm, 1] = self.beta_param[pulled_arm, 1] + 1.0 - reward
        self.t += 1


class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, arms, window_size):
        super().__init__(n_arms, arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(0, self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            if n_samples != 0:
                cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:])
            else:
                cum_rew = 0
            self.beta_parameters[arm, 0] = cum_rew + 1.0
            self.beta_parameters[arm, 1] = n_samples - cum_rew + 1.0


class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        p = self.probabilities[pulled_arm]

        return np.random.binomial(1, p)


class Non_Stationary_Environment(Environment):

    def __init__(self, n_arms, probabilities, horizon, n_phases):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon
        self.n_phases = n_phases

    def round(self, pulled_arm):

        n_phases = self.n_phases

        phase_size = self.horizon / n_phases
        current_phase = int(self.t / phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        return np.random.binomial(1, p)
