from types import SimpleNamespace
from unittest.mock import patch

import pytest

from runtime.gpu_lease import GPUNotIdle, _gpu_busy_reason


@pytest.mark.parametrize("used,gpu,memory,processes,busy", [
    (0, 0, 0, "", False),
    (1, 0, 0, "", True),
    (0, 1, 0, "", True),
    (0, 0, 1, "", True),
    (0, 0, 0, "<process_info><pid>123</pid></process_info>", True),
    (6623, 0, 0, "<process_info><pid>2601241</pid></process_info>", True),
    (0, 0, 0, "Unidentified processes found", None),
])
def test_trimul_requires_an_empty_gpu(used, gpu, memory, processes, busy):
    output = f"""
    <nvidia_smi_log><gpu>
      <utilization><gpu_util>{gpu} %</gpu_util><memory_util>{memory} %</memory_util></utilization>
      <fb_memory_usage><used>{used} MiB</used><reserved>300 MiB</reserved></fb_memory_usage>
      <processes>{processes}</processes>
    </gpu></nvidia_smi_log>
    """
    with patch("runtime.gpu_lease.subprocess.run", return_value=SimpleNamespace(stdout=output)) as query:
        if busy is None:
            with pytest.raises(GPUNotIdle, match="process list is unavailable"):
                _gpu_busy_reason(6)
        else:
            assert bool(_gpu_busy_reason(6)) is busy
        assert query.call_args.args[0][-2:] == ["-i", "6"]
