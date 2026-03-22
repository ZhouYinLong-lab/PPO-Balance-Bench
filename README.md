# 🦭 PPO-Balance-Bench (基于 PPO 的平衡基准测试台)

这是一个基于 **PyBullet** 物理引擎和 **Stable-Baselines3 (PPO)** 算法开发的 3D 强化学习环境。

## 🌟 项目亮点
- **自定义环境**: 完全符合 Gymnasium 接口标准。
- **物理特性**: 模拟了极低摩擦力（0.1）下的物体平衡挑战。
- **PPO 算法**: 使用近端策略优化算法，实现了从随机抽搐到绝对平稳的进化。

## 🚀 快速开始
1. 安装依赖：`pip install "stable-baselines3[extra]" pybullet gymnasium`
2. 训练模型：`python train.py`
3. 观看演示：`python enjoy.py`