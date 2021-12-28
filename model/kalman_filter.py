import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class KalmanFilter:
    def __init__(self, dim, init_transition, init_observe, init_trans_noise, init_obs_noise):
        self.dim = dim
        self.transition_matrix = init_transition
        self.observe_matrix = init_observe
        self.trans_noise = init_trans_noise
        self.obs_noise = init_obs_noise

    def forward(self, data, mu_zero, p_zero):
        time = len(data)

        mu = [mu_zero]
        covar = []
        P = [p_zero]
        c = []
        for t in range(time):
            estimated_noise = self.observe_matrix @ P[-1] @ self.observe_matrix.T + self.obs_noise
            gain = P[-1] @ self.observe_matrix.T @ np.linalg.inv(estimated_noise)

            if t == 0:
                mu.append(mu[0] + gain @ (data[t] - self.observe_matrix @ mu[0]))
                c.append(scipy.stats.multivariate_normal.pdf(data[0], self.observe_matrix@mu[0], estimated_noise))
            else:
                mu.append(self.transition_matrix @ mu[-1]
                          + gain @ (data[t] - self.observe_matrix @ self.transition_matrix @ mu[-1]))
                c.append(scipy.stats.multivariate_normal.pdf(data[t], self.observe_matrix@self.transition_matrix@mu[-2],
                                                             estimated_noise))

            covar.append((np.identity(self.dim) - gain @ self.observe_matrix) @ P[-1])
            P.append(self.transition_matrix@covar[-1]@self.transition_matrix.T + self.trans_noise)

        return np.stack(mu), np.stack(covar), np.array(c)



