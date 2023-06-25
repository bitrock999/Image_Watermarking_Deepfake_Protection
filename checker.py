import pynvml

def get_active_gpus():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    active_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if gpu_info.gpu > 0:
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            active_gpus.append((i, gpu_name))

    pynvml.nvmlShutdown()
    return active_gpus

active_gpus = get_active_gpus()
print("Active GPUs:")
for gpu_index, gpu_name in active_gpus:
    print(f"GPU {gpu_index}: {gpu_name}")
