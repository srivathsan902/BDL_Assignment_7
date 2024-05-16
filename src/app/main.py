from prometheus_client import Counter, Gauge, start_http_server
import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from prometheus_fastapi_instrumentator import Instrumentator
import psutil
from fastapi import Request
import os
import time

# Create an instance of the FastAPI class
app = FastAPI()

# Instrument the FastAPI application
Instrumentator().instrument(app).expose(app)

# Prometheus Metrics
REQUEST_COUNTER = Counter('api_requests_total', 'Total number of API requests', ['client_ip'])

RUN_TIME_GAUGE = Gauge('api_run_time_seconds', 'Running time of the API')
TL_TIME_GAUGE = Gauge('api_tl_time_microseconds', 'Effective processing time per character')

MEMORY_USAGE_GAUGE = Gauge('api_memory_usage', 'Memory usage of the API process')
CPU_USAGE_GAUGE = Gauge('api_cpu_usage_percent', 'CPU usage of the API process')

NETWORK_BYTES_SENT_GAUGE = Gauge('api_network_bytes_sent', 'Network bytes sent by the API process')
NETWORK_BYTES_RECV_GAUGE = Gauge('api_network_bytes_received', 'Network bytes received by the API process')

def format_image(image):
    'Convert the given image of arbitrary size to 28*28 grayscale image'
    resized_image = image.resize((28, 28)).convert("L")
    image_array = np.array(resized_image)
    flattened_image = image_array.flatten()
    return flattened_image

def predict_digit(data_point):
    if data_point.size != (28, 28):
        data_point = format_image(data_point).reshape((1, 784))
    else:
        data_point = data_point.convert("L").reshape((1, 784))
    
    # Placeholder for actual prediction logic
    # time.sleep(2*np.random.random())  # Simulating processing time

    return str(1)  # Placeholder for predicted digit

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

@app.post("/predict/")
async def predict_image(request: Request, file: UploadFile = File(...)):

    start_time = time.time()
    memory_usage_start = process_memory()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Get client's IP address
    client_ip = request.client.host
    
    # Update network I/O gauges
    network_io_counters = psutil.net_io_counters()
    NETWORK_BYTES_SENT_GAUGE.set(network_io_counters.bytes_sent)
    NETWORK_BYTES_RECV_GAUGE.set(network_io_counters.bytes_recv)

    predicted_digit = predict_digit(image)

    cpu_percent = psutil.cpu_percent()
    memory_usage_end = process_memory()
    CPU_USAGE_GAUGE.set(cpu_percent)
    # MEMORY_USAGE_GAUGE.set(memory_usage_start)
    MEMORY_USAGE_GAUGE.set((memory_usage_end-memory_usage_start)/(1024*1024))
    
    # Calculate API running time
    end_time = time.time()
    run_time = end_time - start_time
    
    # Record API usage metrics
    REQUEST_COUNTER.labels(client_ip).inc()
    RUN_TIME_GAUGE.set(run_time)
    
    # Calculate T/L time
    input_length = len(contents)
    tl_time = (run_time / input_length) * 1e6  # microseconds per character
    TL_TIME_GAUGE.set(tl_time)
    
    return {"digit": predicted_digit}

# cpu_percent = psutil.cpu_percent()
# memory_usage = psutil.virtual_memory().used/(1024*1024)
# CPU_USAGE_GAUGE.set(cpu_percent)
# MEMORY_USAGE_GAUGE.set(memory_usage)
if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        reload=True,
        workers=1,
        host="0.0.0.0",
        port=8002
    )
