# ClipSummary

AI-powered video summarization tool that generates summaries and translations from YouTube videos and uploaded files.

## Features

- 🎥 **YouTube & File Upload** - Process videos from URLs or local files with chunked upload support
- 🤖 **AI Summarization** - Generate summaries using BART models
- 🌐 **Multi-language Translation** - Support for Korean and Chinese using NLLB-200
- 📝 **Speech-to-Text** - Accurate transcription with WhisperX
- ⏱️ **Synchronized Playback** - Video player with timestamped subtitles
- 👤 **User Authentication** - JWT-based authentication with user profiles
- 📊 **Video Management** - Track upload status, view history, and manage videos
- 🔄 **Background Processing** - Asynchronous video processing with status tracking
- 💾 **Caching** - Redis-based caching for improved performance

## Quick Start

```bash
# Clone and start
git clone <repository-url>
cd clipSummary
docker-compose up
```

Visit `http://localhost` to use the application.

## Tech Stack

### Frontend
- **Core**: Vanilla JavaScript, HTML5, CSS3
- **UI**: Custom styling with responsive design
- **Video Player**: Custom video player with subtitle synchronization

### Backend
- **Framework**: FastAPI with async support
- **AI/ML**: WhisperX (speech-to-text), BART (summarization), NLLB-200 (translation)
- **Authentication**: JWT tokens with OAuth2 password flow
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis for performance optimization
- **File Processing**: FFmpeg for video processing

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Reverse Proxy**: Nginx with SSL support
- **CI/CD**: GitHub Actions
- **GPU Support**: NVIDIA GPU acceleration for AI models

## Project Structure

```
clipSummary/
├── frontend/                # Frontend application
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript modules
│   ├── images/             # Static assets
│   └── *.html              # HTML pages
├── backend/                # FastAPI backend
│   ├── ai/                 # AI modules (WhisperX, BART, NLLB)
│   ├── api/                # API routes and endpoints
│   ├── db/                 # Database models and connections
│   ├── schemas/            # Pydantic schemas
│   ├── security/           # Authentication logic
│   ├── utils/              # Helper utilities
│   └── tests/              # Test suites
├── nginx/                  # Nginx configuration
├── ssl/                    # SSL certificates
├── uploads/                # User uploaded files
└── docker-compose.yml      # Docker services configuration
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info
- `GET /api/auth/my-videos` - Get user's videos

### Video Processing
- `POST /api/upload/file` - Upload video file
- `POST /api/upload/youtube` - Process YouTube URL
- `POST /api/upload/init-chunked` - Initialize chunked upload
- `POST /api/upload/chunk` - Upload file chunk
- `POST /api/upload/finalize-chunked` - Finalize chunked upload

### Video Management
- `GET /api/videos/{video_id}` - Get video details
- `GET /api/videos/{video_id}/status` - Get processing status
- `DELETE /api/users/me/videos/{video_id}` - Delete user video

## Features in Detail

### Video Processing Pipeline
1. **Upload/URL Processing** - Supports direct file upload or YouTube URL
2. **Audio Extraction** - FFmpeg extracts audio from video
3. **Transcription** - WhisperX generates accurate transcripts
4. **Summarization** - BART model creates concise summaries
5. **Translation** - NLLB-200 translates to Korean/Chinese
6. **Result Caching** - Redis caches results for faster retrieval

### User System
- User registration and authentication
- Personal video library
- Upload history tracking
- Video status monitoring

### Performance Optimizations
- Chunked file uploads for large videos
- Background processing with status updates
- Redis caching for AI model results
- GPU acceleration for AI inference
- Asynchronous request handling

## Development

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Node.js (for frontend development)
- NVIDIA GPU (optional, for acceleration)

### Environment Setup
Create a `.env` file with required environment variables:
```env
JWT_SECRET=your-jwt-secret
POSTGRES_DB=clipsummary
POSTGRES_USER=your-user
POSTGRES_PASSWORD=your-password
REDIS_URL=redis://redis:6379
```

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
```

## License

MIT License
