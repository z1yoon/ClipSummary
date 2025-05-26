# ClipSummary

AI-powered video summarization tool that generates summaries and translations from YouTube videos and uploaded files.

## Features

- 🎥 **YouTube & File Upload** - Process videos from URLs or local files
- 🤖 **AI Summarization** - Generate summaries using BART models
- 🌐 **Multi-language Translation** - Support for Korean and Chinese
- 📝 **Speech-to-Text** - Accurate transcription with WhisperX
- ⏱️ **Synchronized Playback** - Video player with timestamped subtitles

## Quick Start

```bash
# Clone and start
git clone <repository-url>
cd clipSummary
docker-compose up
```

Visit `http://localhost` to use the application.

## Tech Stack

- **Frontend**: React.js, Tailwind CSS
- **Backend**: FastAPI, WhisperX, BART, NLLB-200
- **Storage**: Redis, PostgreSQL
- **Infrastructure**: Docker, GitHub Actions

## Project Structure

```
clipSummary/
├── frontend/           # React frontend
├── backend/            # FastAPI backend
├── nginx/              # Reverse proxy
└── docker-compose.yml  # Docker setup
```

## License

MIT License
