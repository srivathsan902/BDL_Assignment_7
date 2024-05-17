import psutil
import numpy as np

cpu_percent = psutil.cpu_percent(interval =1)
memory_usage_start = psutil.virtual_memory().used/(1024)
a = 3
for i in range(1000000):
    a = a + 1

memory_usage_end = psutil.virtual_memory().used/(1024)
memory_usage = np.abs(memory_usage_end-memory_usage_start)

print(cpu_percent)
print(memory_usage_start)
print(memory_usage)
print(np.random.randint(10))
