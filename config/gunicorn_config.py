import os

# Gunicorn configuration tailored for 3D volumetric MRI processing

# Which port/socket to bind
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")

# Worker Strategy
# Eventlet or Gevent are great for I/O, but sync is better for heavy CPU/GPU PyTorch compute without memory blow-ups
worker_class = "sync"

# Use exactly 1 worker to ensure the huge PyTorch 3D models don't multiply in memory and crash the server
workers = 1

# Long timeout (300 seconds) because 3D TTA inferences take longer than standard 2D web requests
timeout = 300

# To prevent memory leaks in Python
max_requests = 10  # Restart worker after 10 requests to release GPU/RAM
max_requests_jitter = 2