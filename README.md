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

## 🚀 CI/CD Pipeline

This project uses a comprehensive CI/CD pipeline with GitHub Actions that follows a proper development workflow:

### Branch Strategy

- **Feature Branches** → **Dev Branch** → **Main Branch**
- Feature branches are created from `dev`
- All features must be tested before merging to `dev`
- Only `dev` branch can be merged to `main` for production deployment

### Automated Workflows

#### 1. Feature Branch Testing (`feature-to-dev-tests.yml`)
**Triggers:** Pull requests to `dev` branch
**Purpose:** Comprehensive testing before merging features

**Jobs:**
- **Unit Tests** - Backend Python tests with coverage reporting
- **Frontend Tests** - JavaScript syntax checking and linting
- **Integration Tests** - Full stack testing with PostgreSQL and Redis
- **Docker Build Test** - Verify all containers build successfully

**Requirements:** All tests must pass before merge is allowed

#### 2. Dev Branch Validation (`unit-tests.yml` & `integration-tests.yml`)
**Triggers:** Pushes to `dev` branch
**Purpose:** Continuous validation of the development branch

**Jobs:**
- **Unit Tests** - Quick feedback on code quality
- **Integration Tests** - Full system testing with external services

#### 3. Production Deployment (`build-deploy.yml`)
**Triggers:** Pushes to `main` branch (when dev merges to main)
**Purpose:** Deploy to production server

**Features:**
- Secure deployment via Tailscale VPN
- Zero-downtime deployment with health checks
- Automatic rollback on failure
- Docker image cleanup and optimization

### Test Coverage

- **Unit Tests:** Fast, isolated tests for individual components
- **Integration Tests:** End-to-end testing with real databases and services
- **Docker Tests:** Ensure all containers build and run correctly
- **Code Coverage:** Automatic coverage reporting with Codecov

### Development Workflow

1. **Create Feature Branch:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test Locally:**
   ```bash
   # Run tests locally before pushing
   cd backend
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   ```

3. **Create Pull Request:**
   - Push feature branch to GitHub
   - Create PR from `feature/your-feature-name` → `dev`
   - Wait for all CI checks to pass
   - Request code review

4. **Merge to Dev:**
   - After PR approval and all tests pass
   - Merge to `dev` branch
   - Additional validation tests run automatically

5. **Deploy to Production:**
   - Create PR from `dev` → `main`
   - After approval, merge triggers automatic deployment
   - Monitor deployment logs and health checks

### Environment Setup

The CI/CD pipeline requires these GitHub secrets:

```
TAILSCALE_OAUTH_CLIENT_ID    # Tailscale VPN access
TAILSCALE_OAUTH_SECRET       # Tailscale VPN secret
DEPLOY_SSH_KEY              # SSH private key for server access
DEPLOY_USER                 # SSH username for deployment
TAILSCALE_SERVER_IP         # Target server IP address
DEPLOY_DIR                  # Deployment directory on server
```

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd clipSummary

# Start the application using Docker Compose
docker-compose up
```

Visit `http://localhost:3000` to access the application.

## License

[MIT License](LICENSE)
