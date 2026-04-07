# Docker Build and Run Instructions
# Smart Hospital Resource Orchestration Environment

## Quick Start

### Build the Docker Image
```bash
# Build with default settings
docker build -t smart-hospital-orchestration .

# Build with build arguments
docker build \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t smart-hospital-orchestration:1.0.0 .
```

### Run the Container
```bash
# Run with default settings (medium task, 1 episode)
docker run --rm smart-hospital-orchestration

# Run with custom parameters
docker run --rm \
  -e PYTHONPATH=/app \
  smart-hospital-orchestration \
  --task hard --max-steps 100 --verbose

# Run with volume mounts for data persistence
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  smart-hospital-orchestration \
  --task medium --max-steps 75
```

## Advanced Usage

### Interactive Mode
```bash
# Run container in interactive mode for debugging
docker run --rm -it smart-hospital-orchestration bash

# Once inside container:
python -m smart_hospital_orchestration.inference.baseline_inference --help
```

### Development Mode
```bash
# Mount source code for development
docker run --rm -it \
  -v $(pwd):/app \
  smart-hospital-orchestration \
  bash

# Run tests inside container
docker run --rm \
  -v $(pwd):/app \
  smart-hospital-orchestration \
  python -m pytest tests/ -v
```

### Production Mode
```bash
# Run with resource limits
docker run --rm \
  --memory=1g \
  --cpus=0.5 \
  --name hospital-inference \
  smart-hospital-orchestration \
  --task hard --max-steps 100

# Run in detached mode
docker run -d \
  --name hospital-inference \
  --restart unless-stopped \
  smart-hospital-orchestration
```

## Docker Compose (Optional)
```yaml
# docker-compose.yml
version: '3.8'
services:
  hospital-orchestration:
    build: .
    image: smart-hospital-orchestration
    container_name: hospital-inference
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
    command: ["--task", "medium", "--max-steps", "75", "--verbose"]
    restart: unless-stopped
```

```bash
# Run with Docker Compose
docker-compose up --build
docker-compose logs -f
```

## Health Checks
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Inspect health check details
docker inspect --format='{{json .State.Health}}' hospital-inference
```

## Troubleshooting

### Common Issues and Solutions

0. **Docker daemon not running (Windows)**
  ```powershell
  # Start Docker Desktop
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

  # Verify daemon connectivity
  docker version

  # End-to-end local verification (daemon + build + run)
  .\verify_docker.ps1
  ```

1. **Permission Denied**
   ```bash
   # Fix permission issues
   sudo chown -R $USER:$USER ./logs ./data
   ```

2. **Build Cache Issues**
   ```bash
   # Clean build cache
   docker builder prune -f
   docker system prune -f
   ```

3. **Import Errors**
   ```bash
   # Check if package is installed
   docker run --rm smart-hospital-orchestration python -c "import smart_hospital_orchestration"
   ```

4. **Memory Issues**
   ```bash
   # Run with increased memory
   docker run --rm --memory=2g smart-hospital-orchestration
   ```

## Performance Tips

### Optimize Build Time
```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t smart-hospital-orchestration .
```

### Reduce Image Size
```bash
# Use multi-stage build (already implemented)
# Use .dockerignore to exclude unnecessary files
# Use slim base images (already implemented)
```

### Monitoring
```bash
# Monitor resource usage
docker stats hospital-inference

# View logs in real-time
docker logs -f hospital-inference
```
