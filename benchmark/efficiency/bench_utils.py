import torch
import torch.utils.benchmark as benchmark


def benchmark_forward_with_warmup(
    fn,
    *inputs,
    warmups: int = 5,
    repeats: int = 10,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    **kwinputs,
):
    for _ in range(warmups):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    torch.cuda.synchronize()

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    timer = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    return timer, timer.timeit(repeats)
