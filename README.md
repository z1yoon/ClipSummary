# ClipSummary

A production-level application that allows users to summarize YouTube videos and local video files with multi-language subtitles.

## Features

- **Search Videos**: Search by keyword or YouTube URL
- **Upload Video**: Support for MP4/MKV/WebM files
- **AI Summarization**: Using DistilBART/BART for English summaries
- **Translation**: NLLB-200 for Korean/Chinese translations
- **Speech-to-Text**: WhisperX for accurate word-level timestamps
- **Export Summary**: PDF/text export options
- **Synchronized UI**: Video player with synced subtitles in multiple languages

## Tech Stack

### Frontend
- React.js with Vite
- Video.js for video playback
- Tailwind CSS for styling

### Backend
- FastAPI (Python)
- WhisperX for speech-to-text with timestamps
- DistilBART/BART for summarization
- NLLB-200 for translation
- Redis for caching

### Infrastructure
- Docker for containerization
- GitHub Actions for CI/CD
- Azure Container Apps for hosting

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Node.js (v16+)
- Python (v3.9+)

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd clipSummary

# Start the application using Docker Compose
docker-compose up
```

Visit `http://localhost:3000` to access the application.

## Project Structure
```
clipSummary/
├── frontend/           # React frontend
├── backend/            # FastAPI backend
├── .github/workflows/  # CI/CD configuration
├── docker-compose.yml  # Local development setup
└── README.md           # Project documentation
```

## Deployment

The application is configured for deployment on Azure Container Apps using GitHub Actions for CI/CD.


