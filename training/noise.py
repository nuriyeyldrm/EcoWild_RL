import numpy as np


class AnnealedNormalActionNoise:
    def __init__(
        self,
        mean,
        sigma_start,
        sigma_end,
        total_timesteps,
        target_decay_step=30000,
        decay_type="linear",
    ):
        self.mean = mean
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.total_timesteps = total_timesteps
        self.target_decay_step = target_decay_step
        self.decay_type = decay_type
        self.step = 0

    def __call__(self):
        if self.decay_type == "linear":
            frac = max(1 - self.step / self.target_decay_step, 0)
        elif self.decay_type == "fast":
            frac = max(np.exp(-5 * self.step / self.target_decay_step), 0)
        else:
            raise ValueError("Unknown decay_type. Use 'linear' or 'fast'.")

        sigma = self.sigma_end + (self.sigma_start - self.sigma_end) * frac
        self.step += 1
        return np.random.normal(loc=self.mean, scale=sigma, size=self.mean.shape)

    def reset(self):
        self.step = 0