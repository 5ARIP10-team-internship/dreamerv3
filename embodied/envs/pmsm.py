import numpy as np
import scipy.signal as signal
import gym
from gym.spaces import Box
import matplotlib.pyplot as plt
from .utils import SSAnalysis


class EnvPMSM(gym.Env):
    def __init__(self, sys_params, render_mode=None):
        # 系统参数
        self.dt         = sys_params["dt"]
        self.r          = sys_params["r"]
        self.ld         = sys_params["ld"]
        self.lq         = sys_params["lq"]
        self.lambda_PM  = sys_params["lambda_PM"]
        self.we_nom     = sys_params["we_nom"]
        vdc             = sys_params["vdc"]

        # 奖励函数类型
        self.reward_function = sys_params["reward"]

        # 最大电压、电流
        self.vdq_max = vdc / 2
        self.i_max   = sys_params["i_max"]

        # 稳态分析工具
        self.ss_analysis = SSAnalysis()

        # 定义动作和观测空间
        self.low_actions  = np.array([-1.0, -1.0], dtype=np.float32)
        self.high_actions = np.array([ 1.0,  1.0], dtype=np.float32)
        self.action_space = Box(low=self.low_actions,
                                high=self.high_actions,
                                shape=(2,),
                                dtype=np.float32)
        self.action_space.discrete = False

        self.low_observations  = np.array([-1.0]*7, dtype=np.float32)
        self.high_observations = np.array([ 1.0]*7, dtype=np.float32)
        self.observation_space = Box(low=self.low_observations,
                                     high=self.high_observations,
                                     shape=(7,),
                                     dtype=np.float32)
        self.observation_space.discrete = False

        self.render_mode = render_mode

    def step(self, action: np.ndarray):
        # Denormalize
        action_vdq = self.vdq_max * action
        # 线性动态
        s_t = np.array([self.id, self.iq])
        id_next, iq_next = self.ad @ s_t + self.bd @ (action_vdq) + self.wd
        # 限幅
        mag_next = np.linalg.norm([id_next, iq_next])
        if mag_next > self.i_max:
            id_next, iq_next = (self.i_max / mag_next) * np.array([id_next, iq_next])
        # 归一化观测
        obs = np.array([
            id_next / self.i_max,
            iq_next / self.i_max,
            self.id_ref / self.i_max,
            self.iq_ref / self.i_max,
            self.we     / self.we_nom,
            self.prev_vd / self.vdq_max,
            self.prev_vq / self.vdq_max,
        ], dtype=np.float32)

        # 计算奖励
        id_err = abs(self.id/self.i_max - self.id_ref/self.i_max)
        iq_err = abs(self.iq/self.i_max - self.iq_ref/self.i_max)
        dvd    = abs(action[0] - self.prev_vd/self.vdq_max)
        dvq    = abs(action[1] - self.prev_vq/self.vdq_max)

        if   self.reward_function == "absolute":
            reward = -(id_err + iq_err + 0.1*(dvd + dvq))
        elif self.reward_function == "quadratic":
            reward = -((id_err**2 + iq_err**2) + 0.1*(dvd**2 + dvq**2))
        else:
            # 按需添加其他 reward 类型
            reward = -(id_err + iq_err + 0.1*(dvd + dvq))

        # 更新内部状态
        self.id     = id_next
        self.iq     = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        terminated = False
        return obs, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化
        low, high = -0.9, 0.9
        id0_norm     = self.np_random.uniform(low, high)
        iq0_norm     = self.np_random.uniform(low, high)
        id_ref_norm  = self.np_random.uniform(low, high)
        iq_ref_norm  = self.np_random.uniform(low, high)
        we_norm      = self.np_random.uniform(0, high)

        # 设定状态空间
        self.id     = id0_norm   * self.i_max
        self.iq     = iq0_norm   * self.i_max
        self.id_ref = id_ref_norm* self.i_max
        self.iq_ref = iq_ref_norm* self.i_max
        self.we     = we_norm    * self.we_nom
        self.prev_vd = 0.0
        self.prev_vq = 0.0

        # 离散化连续系统
        a = np.array([
            [-self.r/self.ld,           self.we*self.lq/self.ld],
            [-self.we*self.ld/self.lq,  -self.r/self.lq]
        ])
        b = np.array([[1/self.ld, 0],[0, 1/self.lq]])
        w = np.array([[0],[-self.we*self.lambda_PM]])
        bw = np.hstack((b, w))
        c  = np.eye(2)
        d  = np.zeros((2,1))
        ad, bwd, _, _, _ = signal.cont2discrete((a, bw, c, d), self.dt, method='zoh')
        self.ad = ad
        self.bd = bwd[:,:2]
        self.wd = bwd[:,2].squeeze()

        # 返回初始观测
        obs = np.array([
            self.id   / self.i_max,
            self.iq   / self.i_max,
            self.id_ref/self.i_max,
            self.iq_ref/self.i_max,
            self.we   / self.we_nom,
            0.0, 0.0
        ], dtype=np.float32)
        return obs, {}


def PMSM(sys_params=None, render_mode=None, **kwargs):
    # 兼容 DreamerV3 make_env 传入的 task
    if not isinstance(sys_params, dict):
        sys_params = {
            "dt": 1/10e3,
            "r": 29.0808e-3,
            "ld": 0.91e-3,
            "lq": 1.17e-3,
            "lambda_PM": 0.172312604,
            "vdc": 1200,
            "we_nom": 200*2*np.pi,
            "i_max": 200,
            "reward": "absolute",
        }
    return EnvPMSM(sys_params, render_mode=render_mode, **kwargs)


# 注册标准 Gym 环境，最大步长 200
gym.register(
    id='PMSM-v0',
    entry_point='embodied.envs.pmsm:PMSM',
    max_episode_steps=200,
)
