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


class Environment():
    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        p = self.probabilities[pulled_arm]

        return np.random.binomial(1, p)


class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        n_phases = len(self.probabilities)
        phase_size = self.horizon / n_phases
        current_phase = int(self.t / phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        return np.random.binomial(1, p)
