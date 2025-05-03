from embodied.envs.dmc import DMC
from embodied.envs.pmsm import EnvPMSM

def load(name, config):
    if name == 'dmc':
        return DMC(config)
    if name == 'pmsm':
        return EnvPMSM(config.get("sys_params", {}))
    raise NotImplementedError(name)
