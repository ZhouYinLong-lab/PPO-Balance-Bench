import time
from stable_baselines3 import PPO
from seal_env_physics import SealBalanceEnv
if __name__ == "__main__":
    env = SealBalanceEnv()
    model = PPO.load("ppo_seal_brain")
    obs, info = env.reset(seed=42)
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()