import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium.utils import seeding


class SealBalanceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space=spaces.Box(low=-0.2,high=0.2,shape=(2,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(6,),dtype=np.float32)
        self.physicsClient=None
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if self.physicsClient is None:
            self.physicsClient = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.planeId = p.loadURDF("plane.urdf")
        platform_half_extents = [0.3, 0.3, 0.05]
        platform_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=platform_half_extents)
        platform_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=platform_half_extents,
                                              rgbaColor=[0.1, 0.5, 0.9, 1])
        self.platform_id = p.createMultiBody(
            baseMass=1000,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0, 0, 0.5]
        )
        ball_radius = 0.1
        ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0.2, 0.2, 1])
        rand_x = self.np_random.uniform(low=-0.05, high=0.05)
        rand_y = self.np_random.uniform(low=-0.05, high=0.05)
        self.ball_start_pos = [rand_x, rand_y, 0.5 + platform_half_extents[2] + ball_radius + 0.05]

        self.ball_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=self.ball_start_pos
        )
        p.changeDynamics(self.ball_id, -1, lateralFriction=0.1)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        _, plat_ori = p.getBasePositionAndOrientation(self.platform_id)
        plat_euler = p.getEulerFromQuaternion(plat_ori)
        obs = np.array([
            ball_pos[0],  # [0] 球的 X 坐标
            ball_pos[1],  # [1] 球的 Y 坐标
            ball_vel[0],  # [2] 球在 X 轴的滚动速度
            ball_vel[1],  # [3] 球在 Y 轴的滚动速度
            plat_euler[0],  # [4] 平台的 Roll (左右倾斜角)
            plat_euler[1]  # [5] 平台的 Pitch (前后倾斜角)
        ], dtype=np.float32)
        return obs, {}
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_roll = float(action[0])
        target_pitch = float(action[1])
        target_quat = p.getQuaternionFromEuler([target_roll, target_pitch, 0])
        p.resetBasePositionAndOrientation(
            self.platform_id,
            posObj=[0, 0, 0.5],
            ornObj=target_quat
        )
        p.stepSimulation()
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        _, plat_ori = p.getBasePositionAndOrientation(self.platform_id)
        plat_euler = p.getEulerFromQuaternion(plat_ori)
        obs = np.array([
            ball_pos[0], ball_pos[1],
            ball_vel[0], ball_vel[1],
            plat_euler[0], plat_euler[1]
        ], dtype=np.float32)
        terminated = bool(
            abs(ball_pos[0]) > 0.3 or
            abs(ball_pos[1]) > 0.3 or
            ball_pos[2] < 0.4
        )
        if terminated:
            reward = -10.0
        else:
            dist = float(np.sqrt(ball_pos[0] ** 2 + ball_pos[1] ** 2))
            reward = 1.0 + (0.3 - dist)
        truncated = False
        return obs, reward, terminated, truncated, {}