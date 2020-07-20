import numpy as np


class Learner():
    def __init__(self, n_arms, arms):
        self.n_arms = n_arms
        self.arms = arms
        self.t = 0
        self.rewards = []
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.pulled_arms = []

    def update_observation(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.pulled_arms.append(pulled_arm)
        self.rewards.append(reward)


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
    def __init__(self, n_arms, arms, window_size, horizon):
        super().__init__(n_arms, arms)
        self.window_size = window_size
        self.day = 0
        self.pulled_arms_per_day = np.zeros([horizon, n_arms])

    def update(self, pulled_arm, reward, day):
        self.t += 1
        self.update_observation(pulled_arm, reward)
        self.pulled_arms_per_day[day, pulled_arm] += 1
        for arm in range(0, self.n_arms):

            if (day + 1 - self.window_size) < 0:
                start_index = 0
            else:
                start_index = day + 1 - self.window_size

            n_samples = int(np.sum(self.pulled_arms_per_day[start_index:(day + 1), arm]))

            if n_samples != 0:
                cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:])
            else:
                cum_rew = 0
            self.beta_param[arm, 0] = cum_rew + 1.0
            self.beta_param[arm, 1] = n_samples - cum_rew + 1.0


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

    def round(self, pulled_arm, day):
        n_phases = self.probabilities.shape[0]

        phase_size = self.horizon / n_phases
        current_phase = int(day / phase_size)
        p = self.probabilities[current_phase, pulled_arm]
        self.t += 1

        return np.random.binomial(1, p)

    def update_observation(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.rewards.append(reward)
