import functools

import elements
import embodied
import gym
import numpy as np


class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    # 如果要 reset
    if action.get('reset', False) or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)

    # 解包 action 字典
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]

    # 调用底层 env.step
    result = self._env.step(action)
    # 兼容 Gym 和 Gymnasium 的 4-tuple / 5-tuple 返回
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
      done = bool(terminated or truncated)
    else:
      obs, reward, done, info = result
      done = bool(done)

    self._done = done
    self._info = info or {}

    return self._obs(
      obs, reward,
      is_last=done,
      is_terminal=bool(self._info.get('is_terminal', done))
    )

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    # 如果不是 dict 且是 sequence，取第一个
    if not self._obs_dict and isinstance(obs, (tuple, list)):
      obs = obs[0]
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
      reward=np.float32(reward),
      is_first=is_first,
      is_last=is_last,
      is_terminal=is_terminal
    )
    return obs

  def render(self):
    img = self._env.render('rgb_array')
    assert img is not None
    return img

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      name = f"{prefix}/{key}" if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, name))
      else:
        result[name] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for p in parts[:-1]:
        node = node.setdefault(p, {})
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
