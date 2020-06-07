import numpy as np

class kalman_filter:
    def __init__(self, A, B, H, P, Q, R, xo):
        self.A = A              # state transition matrix --> dynamics
        self.B = B              # control command --> state change
        self.H = H              # observation matrix
        self.P = P              # covariance of state vector estimate
        self.Q = Q              # process noise covariance
        self.R = R              # measurement noise covariance
        self.xo = xo            # intialized state vector

    def estimate_state(self, z, u):
        X_hat = np.matmul(self.A, self.xo) + np.matmul(self.B, u)
        self.P_hat = np.matmul(self.A, np.matmul(self.P, self.A.transpose())) + self.Q
        self.K = np.matmul(self.P_hat, np.matmul(self.H.T, np.linalg.inv(self.R + np.matmul(self.H, np.matmul(self.P_hat, self.H.transpose())))))
        self.xo = self.xo + np.matmul(self.K, (z - np.matmul(self.H, self.xo)))
        self.P = self.P_hat - np.matmul(self.K, np.matmul(self.H, self.P_hat))
        return self.xo
