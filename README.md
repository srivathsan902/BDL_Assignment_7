# Health Monitoring of FastAPI

## Overview
FastAPI Health Monitoring through Prometheus web scraping for metrics and Grafana for visualization of the metrics. 

## Installation
Install Docker and WSL to your local computer.

Run the command `docker-compose up --build` to start the FastAPI application and host the Grafana UI

## Walkthrough of Code
The main.py is present in the `root/src/app/` directory. It contains the FastAPI application code, with relevant metrics updated to Prometheus.
The `prometheus.yml` file is present inside the `root/prometheus_data` directory. Place the `Dockerfile` and `requirements.txt` file in the `src` directory and the `docker-compose.yml` file in the `root/` directory.

* The main.py file contains the following metrics posted to Prometheus
    1. API Usage Counters: Number of hits from different client IP addresses.
    2. API Run Time: Total time taken by the API to process requests.
    3. API Processing Time (T/L): Effective processing time per character.
    4. CPU Utilization: Rate of CPU usage by the FastAPI application.
    5. Memory Utilization: Memory used by the FastAPI application.
    6. Network I/O: Bytes sent and received by the application.
    7. Network I/O Rate: Rate of bytes sent and received by the application 

    These can be queried in Grafana and visualized

## Additional Information
In case you add some code on top of this, you may have to add the libraries that you use in the `requirements.txt` file in the. Repeat the command and `docker-compose up --build` command as done earlier.

Once the usage is over, ensure to run the command `docker-compose down` to shutdown the local host.
In case your system consumes a lot of memory and disk usage, then open the command promt with administrator access and run `wsl --shutdown` after this.

## Helpful Links
* [Blog Post on Health Monitoring with Prometheus and Grafana](https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn#:~:text=Access%20Prometheus%20by%20navigating%20to,see%20the%20Prometheus%20web%20interface.&text=This%20step%20automatically%20adds%20Prometheus,to%20apply%20the%20instrumentation%20changes)
* [FastAPI - Swagger UI Port 8002](http://127.0.0.1:8002/docs)
* [FastAPI - Swagger UI Port 8100](http://127.0.0.1:8002/docs)
* [FastAPI - Swagger UI Port 8101](http://127.0.0.1:8002/docs)
* [FastAPI - Swagger UI Port 8102](http://127.0.0.1:8002/docs)
* [Official Documentation for FastAPI](https://fastapi.tiangolo.com/)


