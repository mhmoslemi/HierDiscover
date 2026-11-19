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
        copies = []
        host_bytes = 0
        for parameter in parameters:
            state = self.state.get(parameter, {})
            for key in self._moment_keys:
                moment = state.get(key)
                if moment is None or moment.device.type == "cpu":
                    continue
                host = parked.get((parameter, key))
                if host is None:
                    host_bytes += moment.numel() * moment.element_size()
                copies.append((state, key, moment, host))

        def copy_bucket():
            buffers = [torch.empty_like(moment, device="cpu") if host is None else host
                       for _, _, moment, host in copies]
            streams = set()
            try:
                for (_, _, moment, _), host in zip(copies, buffers):
                    if moment.is_cuda:
                        streams.add(torch.cuda.current_stream(moment.device))
                    host.copy_(moment, non_blocking=moment.is_cuda)
            finally:
                # Publish CPU moments only once the complete bucket is ready.
                for stream in streams:
                    stream.synchronize()
            return buffers

        buffers = (retry_host_allocation(
            copy_bucket, hardware_profile=self._hardware_profile,
            stage="GPT-OSS Adam moments", required_bytes=host_bytes)
            if host_bytes else copy_bucket())
        for (state, key, _, _), host in zip(copies, buffers):
            state[key] = host

    def load_state_dict(self, state_dict):
        # AdamW normally casts state to the live parameter's device. Keep the
        # saved moments on CPU so restarting after eval avoids a CUDA round trip.
        saved_state = state_dict["state"]
        state_without_moments = dict(saved_state)
        moments_by_parameter = {}
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                state = saved_state.get(saved_id, {})
                moments = {key: state[key] for key in self._moment_keys if key in state}
                if moments:
                    moments_by_parameter[parameter] = moments
                    state_without_moments[saved_id] = {
                        key: value for key, value in state.items() if key not in moments}

        host_bytes = sum(moment.numel() * parameter.element_size()
                         for parameter, moments in moments_by_parameter.items()
                         for moment in moments.values())

        def copy_moments():
            return {
                parameter: {key: moment.detach().to(device="cpu", dtype=parameter.dtype, copy=True)
                            for key, moment in moments.items()}
                for parameter, moments in moments_by_parameter.items()
            }

        host_moments = (retry_host_allocation(
            copy_moments, hardware_profile=self._hardware_profile,
            stage="GPT-OSS Adam checkpoint restore", required_bytes=host_bytes)
            if moments_by_parameter else {})
        super().load_state_dict({**state_dict, "state": state_without_moments})
        for parameter, moments in host_moments.items():
            self.state[parameter].update(moments)

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
                                    state[key] = host.to(parameter.device, non_blocking=parameter.is_cuda)
                        self.param_groups = [dict(group, params=bucket)]
                        super().step()
                    finally:
                        self._park_moments(bucket, parked)
        finally:
            self.param_groups = groups
        return loss
