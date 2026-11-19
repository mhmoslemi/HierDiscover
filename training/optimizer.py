"""GPU AdamW with bounded resident optimizer state for the large GPT-OSS adapter."""

import torch

from runtime.host_memory import retry_host_allocation
from training.fast_distributed import parameter_buckets


class HostMomentAdamW(torch.optim.AdamW):
    moments_on_host = True
    _moment_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    def __init__(self, *args, hardware_profile, **kwargs):
        super().__init__(*args, **kwargs)
        self._hardware_profile = hardware_profile

    def _park_moments(self, parameters, parked):
        for parameter in parameters:
            state = self.state.get(parameter, {})
            for key in self._moment_keys:
                moment = state.get(key)
                if moment is None or moment.device.type == "cpu":
                    continue
                host = parked.get((parameter, key))
                if host is None:
                    host = retry_host_allocation(
                        moment.cpu, hardware_profile=self._hardware_profile,
                        stage="GPT-OSS Adam moments", required_bytes=moment.numel() * moment.element_size())
                else:
                    host.copy_(moment)
                state[key] = host

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._park_moments(self.state, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            # AdamW's update is per parameter. Global clipping and gradient
            # aggregation have already run once on the complete batch.
            for group in groups:
                active = (parameter for parameter in group["params"] if parameter.grad is not None)
                for bucket in parameter_buckets(active):
                    parked = {}
                    try:
                        for parameter in bucket:
                            state = self.state.get(parameter, {})
                            for key in self._moment_keys:
                                host = state.get(key)
                                if host is not None and host.device != parameter.device:
                                    parked[parameter, key] = host
                                    state[key] = host.to(parameter.device)
                        self.param_groups = [dict(group, params=bucket)]
                        super().step()
                    finally:
                        self._park_moments(bucket, parked)
        finally:
            self.param_groups = groups
        return loss
