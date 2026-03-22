from stable_baselines3 import PPO
from seal_env_physics import SealBalanceEnv
if __name__ == "__main__":
    env = SealBalanceEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_seal_brain")