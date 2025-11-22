# Smart City AI - Vision & Backend System

A real-time AI system for detecting urban incidents using YOLOv11-Nano and processing them through a scalable backend pipeline.

## 🏗️ Architecture

```
RTSP Stream → YOLOv11 Detection → Frame Buffer → MinIO Storage
                                        ↓
                                   Redis Queue
                                        ↓
                              Celery Worker → PostgreSQL
                                        ↓
                                  FastAPI Backend
                                        ↓
                              WebSocket Dashboard
```

## 🎯 Detected Incident Types

- Accident
- Fire/Smoke
- Fight/Violence
- Flood/Waterlogging
- Garbage/Pollution

## 📋 Prerequisites

### Required Services
- **Python 3.10**
- **PostgreSQL 14+** with PostGIS extension
- **Redis 7+**
- **MinIO** (or AWS S3)
- **CUDA 12.6** (for GPU acceleration)

### Hardware Requirements
- GPU: RTX 3050 6GB (or equivalent)
- RAM: 8GB minimum
- Storage: 20GB for models and frames

## 🚀 Quick Start

### 1. Clone and Setup

```powershell
cd "c:\Users\Thiya\OneDrive\Documents\Smart city AI"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update:

```powershell
cp .env.example .env
```

Edit `.env` with your configuration:
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/smart_city_ai
REDIS_HOST=localhost
REDIS_PORT=6379
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
YOLO_MODEL_PATH=models/yolov11-nano.onnx
STREAM_URL=rtsp://your-camera-stream
```

### 3. Setup Database

```powershell
# Create PostgreSQL database
psql -U postgres
CREATE DATABASE smart_city_ai;
\c smart_city_ai
CREATE EXTENSION postgis;
\q

# Run migrations
alembic upgrade head
```

### 4. Start Services

You need **4 terminal windows**:

**Terminal 1 - Redis:**
```powershell
redis-server
```

**Terminal 2 - MinIO:**
```powershell
minio server C:\minio-data --console-address ":9001"
```

**Terminal 3 - Celery Worker:**
```powershell
cd "c:\Users\Thiya\OneDrive\Documents\Smart city AI"
celery -A tasks.celery_worker worker --loglevel=info --pool=solo
```

**Terminal 4 - FastAPI Backend:**
```powershell
cd "c:\Users\Thiya\OneDrive\Documents\Smart city AI"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 5 - Redis Consumer:**
```powershell
cd "c:\Users\Thiya\OneDrive\Documents\Smart city AI"
python tasks/redis_consumer.py
```

**Terminal 6 - Vision Pipeline:**
```powershell
cd "c:\Users\Thiya\OneDrive\Documents\Smart city AI"
python vision/yolov11_pipeline.py
```

## 📁 Project Structure

```
Smart city AI/
├── backend/
│   └── main.py                 # FastAPI application with REST + WebSocket
├── database/
│   ├── models.py               # SQLAlchemy models
│   └── migrations/             # Alembic migrations
│       ├── env.py
│       ├── script.py.mako
│       └── versions/
│           └── 0001_create_incidents_table.py
├── storage/
│   └── minio_client.py         # MinIO/S3 client
├── tasks/
│   ├── celery_worker.py        # Celery tasks
│   ├── redis_consumer.py       # Redis event consumer
│   └── redis_producer.py       # Redis event publisher
├── vision/
│   └── yolov11_pipeline.py     # YOLOv11 detection + frame buffer
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── alembic.ini                # Alembic configuration
└── .env.example               # Environment template
```

## 🔌 API Endpoints

### REST API
- `POST /incidents/create` - Create new incident
- `GET /incidents/list` - List all incidents
- `GET /incidents/{id}` - Get specific incident
- `POST /verify/from_llm` - Update verification status (for LLM service)
- `GET /health` - Health check

### WebSocket
- `WS /ws` - Real-time incident updates

### Example Usage

```python
import requests

# Create incident
response = requests.post("http://localhost:8000/incidents/create", json={
    "incident": "accident",
    "confidence": 0.87,
    "frame_urls": ["http://minio:9000/frames/img1.jpg"],
    "timestamp": "2025-11-18T12:00:00Z",
    "location": {"lat": 11.0222, "lon": 77.0133}
})

# List incidents
incidents = requests.get("http://localhost:8000/incidents/list").json()

# Verify incident (called by LLM service)
requests.post("http://localhost:8000/verify/from_llm", 
    params={"id": "incident_123", "status": "verified"})
```

## 🎥 YOLOv11 Model Setup

1. Download or train YOLOv11-nano model
2. Export to ONNX format:
```python
from ultralytics import YOLO
model = YOLO('yolov11n.pt')
model.export(format='onnx')
```
3. Place in `models/yolov11-nano.onnx`

## 🔧 Configuration Options

### Vision Pipeline (`config.py`)
- `CONFIDENCE_THRESHOLD` - Minimum detection confidence (0.0-1.0)
- `FRAME_BUFFER_SIZE` - Rolling buffer size (default: 16)
- `FRAMES_TO_EXTRACT` - Frames saved per incident (default: 4)

### Database
- Supports PostGIS for geospatial queries
- Async SQLAlchemy for high performance

### Storage
- MinIO for local/private cloud
- S3-compatible interface

## 🐛 Troubleshooting

### CUDA not found
```powershell
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### PostgreSQL connection error
- Verify PostgreSQL is running: `pg_isready`
- Check credentials in `.env`
- Ensure PostGIS extension is installed

### Redis connection error
- Verify Redis is running: `redis-cli ping`
- Should return `PONG`

### MinIO bucket errors
- Access MinIO console: `http://localhost:9001`
- Verify bucket `frames` exists
- Check access credentials

## 📊 Monitoring

- **FastAPI Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001
- **Celery Flower** (optional):
```powershell
pip install flower
celery -A tasks.celery_worker flower
```

## 🔄 Development Workflow

1. Vision pipeline detects incident → publishes to Redis
2. Redis consumer triggers Celery task
3. Celery worker stores incident in PostgreSQL
4. FastAPI serves data via REST/WebSocket
5. (Optional) LLM service verifies incident

## 📝 Next Steps

1. **Train Custom YOLOv11 Model** - Fine-tune on your urban incident dataset
2. **Integrate LLM Service** - Uncomment LLM tasks in `redis_consumer.py`
3. **Add Dashboard** - Build frontend with WebSocket support
4. **Deploy** - Use Docker Compose for production deployment
5. **Add Monitoring** - Integrate Prometheus + Grafana

## 🐳 Docker Deployment (Coming Soon)

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgis/postgis:14-3.3
  redis:
    image: redis:7-alpine
  minio:
    image: minio/minio
  backend:
    build: .
    command: uvicorn backend.main:app
  celery:
    build: .
    command: celery -A tasks.celery_worker worker
  vision:
    build: .
    command: python vision/yolov11_pipeline.py
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📧 Support

For issues and questions, open a GitHub issue or contact the development team.
# Vision-Guard
