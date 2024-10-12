import psutil
import torch


def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}:")
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (
                1024**3
            )  # Convert bytes to GB
            print(f"  Total GPU Memory: {gpu_mem:.2f} GB")
    else:
        print("No GPU available.")


# Get system RAM info
def print_system_memory():
    ram = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    print(f"Total System RAM: {ram:.2f} GB")


# Get the number of CPU cores
def print_max_workers():
    cpu_count = psutil.cpu_count(
        logical=True
    )  # Logical cores (including hyperthreading)
    print(f"Max CPU Workers (Logical Cores): {cpu_count}")


# Main function
def pc_info():
    print_system_memory()
    print_gpu_memory()
    print_max_workers()
