import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from seal_env_physics import SealBalanceEnv

if __name__ == "__main__":
    print("🚀 正在启动多核并行物理环境...")
    
    # 核心魔法：利用你的多核 CPU，同时开启 8 个物理世界！
    # 相当于同时有 8 个小球在 8 个平台上试错，经验共享给同一个大脑。
    env = make_vec_env(SealBalanceEnv, n_envs=8)
    
    # 初始化 PPO 大脑，开启详细日志
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("🧠 开始极限训练：500,000 步...")
    start_time = time.time()
    
    # 训练步数直接翻 5 倍
    model.learn(total_timesteps=500000)
    
    end_time = time.time()
    print(f"✅ 训练完成！耗时: {(end_time - start_time) / 60:.2f} 分钟")
    
    # 保存你训练出的“超级大脑”
    model.save("ppo_seal_brain_ultra")
    print("💾 模型已保存为 ppo_seal_brain_ultra.zip")