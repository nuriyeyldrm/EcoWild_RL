from stable_baselines3.common.callbacks import BaseCallback
import torch


class ResetPolicyCallback(BaseCallback):
    def __init__(self, reset_every=5, verbose=0):
        super().__init__(verbose)
        self.reset_every = reset_every
        self.episode_count = 0

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        if done_array is not None and any(done_array):
            self.episode_count += 1
            if self.episode_count % self.reset_every == 0:
                print(f"Resetting policy at episode {self.episode_count}")
                self.model.policy.actor.apply(self.model.policy.init_weights)
                self.model.policy.critic.apply(self.model.policy.init_weights)
                self.model.policy.critic_target.load_state_dict(
                    self.model.policy.critic.state_dict()
                )
                self.model.policy.actor_target.load_state_dict(
                    self.model.policy.actor.state_dict()
                )
        return True


class ResetPolicyOnHighSamplingCallback(BaseCallback):
    def __init__(self, threshold=55, required_streak=4, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.required_streak = required_streak
        self.streak_counter = 0

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")

        if done_array is not None and any(done_array):
            # Unwrap environment until we reach base env
            env = self.model.get_env().envs[0]
            while hasattr(env, "env"):
                env = env.env

            if hasattr(env, "avg_sampling_time_list") and len(env.avg_sampling_time_list) > 0:
                avg = env.avg_sampling_time_list[-1]

                if avg >= self.threshold:
                    self.streak_counter += 1
                else:
                    self.streak_counter = 0

                if self.streak_counter >= self.required_streak:
                    print(
                        f"Resetting policy due to {self.streak_counter} high-sampling episodes!"
                    )
                    self.reset_policy_weights()
                    self.streak_counter = 0

        return True

    def reset_policy_weights(self):
        actor = self.model.policy.actor
        critic = self.model.policy.critic
        actor_target = self.model.policy.actor_target
        critic_target = self.model.policy.critic_target

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        actor.apply(init_weights)
        critic.apply(init_weights)
        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())

        print("Policy reset complete.")


class EpisodeModelSaverCallback(BaseCallback):
    def __init__(self, save_path_prefix, verbose=0):
        super().__init__(verbose)
        self.save_path_prefix = save_path_prefix
        self.episode_counter = 0

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        if done_array is not None and any(done_array):
            model_path = f"{self.save_path_prefix}_episode_{self.episode_counter}.zip"
            self.model.save(model_path)
            print(f"Saved model: {model_path}")
            self.episode_counter += 1
        return True


class ResetNoiseCallback(BaseCallback):
    def __init__(self, action_noise, verbose=0):
        super().__init__(verbose)
        self.action_noise = action_noise

    def _on_rollout_start(self) -> None:
        self.action_noise.reset()

    def _on_step(self) -> bool:
        return True # Required abstract method, just returns True to keep training going