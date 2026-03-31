# Docker Setup for Visual Search Engine

This document explains how to build and run the Visual Search Engine using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB of available RAM
- At least 10GB of available disk space

## Quick Start

### 1. Create Environment File

Create a `.env` file in the root directory:

```bash
# Weaviate Configuration
WEAVIATE_URL=http://weaviate:8080
WEAVIATE_API_KEY=your_weaviate_api_key_here

# Flickr API (optional, for scraping images)
FLICKR_KEY=your_flickr_key_here
FLICKR_SECRET=your_flickr_secret_here
```

### 2. Start All Services

```bash
docker-compose up -d
```

This will start:
- **Weaviate** on port 8080
- **FastAPI Backend** on port 8000
- **React Frontend** on port 80

### 3. Access the Application

- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Weaviate Console**: http://localhost:8080/v1/objects

## Docker Compose Commands

### Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Start specific service
docker-compose up backend

# Start with logs
docker-compose up
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend

# Follow logs
docker-compose logs -f backend
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
```

## Building Images

### Build All Images

```bash
docker-compose build
```

### Build Specific Image

```bash
docker-compose build backend
docker-compose build frontend
```

### Rebuild Without Cache

```bash
docker-compose build --no-cache
```

## Individual Service Management

### Backend

```bash
# Build backend image
docker build -f backend/Dockerfile -t visual-search-backend .

# Run backend container
docker run -p 8000:8000 \
  -e WEAVIATE_URL=http://weaviate:8080 \
  -e WEAVIATE_API_KEY=your_key \
  visual-search-backend

# Backend Shell
docker-compose exec backend bash
```

### Frontend

```bash
# Build frontend image
docker build -f frontend/Dockerfile -t visual-search-frontend .

# Run frontend container
docker run -p 80:80 visual-search-frontend

# Frontend Shell
docker-compose exec frontend sh
```

### Weaviate

```bash
# Weaviate Shell
docker-compose exec weaviate bash

# Access Weaviate data
docker-compose exec weaviate ls -la /var/lib/weaviate
```
```

## Development Mode

For development, you might want to mount source code as volumes:

```yaml
# In docker-compose.yml, add:
services:
  backend:
    volumes:
      - ./backend:/app
      - /app/__pycache__
```

## Production Considerations

### Security

1. **Remove anonymous access** to Weaviate in production
2. **Use strong API keys** and secrets
3. **Enable HTTPS** with a reverse proxy (nginx/traefik)
4. **Limit resource usage** with Docker constraints

### Resource Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Persistent Data

Weaviate data is persisted in a Docker volume. To backup:

```bash
# Backup Weaviate data
docker run --rm -v visual-search-engine_weaviate_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/weaviate-backup.tar.gz /data

# Restore Weaviate data
docker run --rm -v visual-search-engine_weaviate_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/weaviate-backup.tar.gz -C /
```

## Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# Check specific service health
docker-compose ps backend
docker-compose ps frontend
docker-compose ps weaviate
```

## Troubleshooting

### Backend Issues

**Problem**: Backend fails to connect to Weaviate

**Solution**:
```bash
# Check if Weaviate is ready
docker-compose logs weaviate

# Restart backend after Weaviate is ready
docker-compose restart backend
```

### Frontend Issues

**Problem**: Frontend can't reach backend API

**Solution**: Check nginx configuration in `frontend/nginx.conf` - ensure API proxy is correctly configured.

### Out of Memory

**Problem**: Services keep restarting

**Solution**:
```bash
# Check resource usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
```

### Build Errors

**Problem**: Build fails with "no space left on device"

**Solution**:
```bash
# Clean up Docker resources
docker system prune -a --volumes
```

## Environment Variables

| Variable | Service | Required | Default |
|----------|---------|----------|---------|
| `WEAVIATE_URL` | Backend | Yes | http://weaviate:8080 |
| `WEAVIATE_API_KEY` | Backend | No | - |
| `FLICKR_KEY` | Backend | No | - |
| `FLICKR_SECRET` | Backend | No | - |

## Network Architecture

```
Internet
    |
    v
[Nginx on Frontend:80] --> [FastAPI Backend:8000] --> [Weaviate:8080]
```

- Frontend and Backend communicate via internal Docker network
- Backend connects to Weaviate via internal network
- Only Frontend port (80) is exposed externally
