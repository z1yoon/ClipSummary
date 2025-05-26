// Video Player with modern UI for ClipSummary
class VideoPlayer {
    constructor(videoElement, container) {
        this.video = videoElement;
        this.container = container;
        this.currentLanguage = 'en';
        this.availableLanguages = ['en'];
        this.subtitles = {};
        this.currentProgress = 0;
        this.summary = '';
        
        // Add event listeners for the video element
        this.addVideoEventListeners();
    }
    
    addVideoEventListeners() {
        // Update subtitle display when time updates
        this.video.addEventListener('timeupdate', () => {
            this.updateSubtitleDisplay(this.video.currentTime);
        });
    }
    
    updateSubtitleDisplay(currentTime) {
        if (!this.subtitles || !this.subtitles[this.currentLanguage]) return;
        
        const subtitleDisplay = document.getElementById('subtitle-display');
        if (!subtitleDisplay) return;
        
        // Find the subtitle that corresponds to current time
        const currentSubtitle = this.subtitles[this.currentLanguage].find(
            sub => currentTime >= sub.start && currentTime <= sub.end
        );
        
        if (currentSubtitle) {
            subtitleDisplay.textContent = currentSubtitle.text;
            subtitleDisplay.style.display = 'block';
        } else {
            subtitleDisplay.textContent = '';
            subtitleDisplay.style.display = 'none';
        }
    }
    
    // Load subtitles for a specific language
    loadSubtitles(language, subtitles) {
        this.subtitles = this.subtitles || {};
        this.subtitles[language] = subtitles;
        this.currentLanguage = language;
        
        // Update subtitle display immediately
        this.updateSubtitleDisplay(this.video.currentTime);
    }
    
    // Show a processing overlay with spinner and status message
    showProcessingOverlay(message) {
        // Remove existing overlay if there is one
        this.hideProcessingOverlay();
        
        // Create processing overlay with modern styling
        const overlay = document.createElement('div');
        overlay.className = 'processing-overlay';
        
        const spinner = document.createElement('div');
        spinner.className = 'processing-spinner';
        
        const messageElement = document.createElement('div');
        messageElement.className = 'processing-text';
        messageElement.textContent = message || 'Processing your video...';
        
        overlay.appendChild(spinner);
        overlay.appendChild(messageElement);
        
        this.container.appendChild(overlay);
        
        this.processingOverlay = {
            element: overlay,
            message: messageElement,
            startTime: Date.now()
        };
        
        // Start polling for processing status
        this.startProcessingStatusPolling();
    }
    
    hideProcessingOverlay() {
        if (this.processingOverlay) {
            this.container.removeChild(this.processingOverlay.element);
            this.processingOverlay = null;
        }
        
        // Stop polling if it's running
        if (this.processingStatusInterval) {
            clearInterval(this.processingStatusInterval);
            this.processingStatusInterval = null;
        }
    }
    
    startProcessingStatusPolling() {
        // Clear existing interval if any
        if (this.processingStatusInterval) {
            clearInterval(this.processingStatusInterval);
        }
        
        const videoId = this.videoId;
        if (!videoId) return;
        
        this.processingStatusInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/upload/status/${videoId}`, {
                    headers: this.getAuthHeaders()
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to get status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Update the processing overlay
                if (this.processingOverlay) {
                    // Update message
                    if (data.message) {
                        this.processingOverlay.message.textContent = data.message;
                    }
                    
                    // If processing is complete, reload the page to get full results
                    if (data.status === 'completed') {
                        clearInterval(this.processingStatusInterval);
                        this.processingStatusInterval = null;
                        
                        // Show success message briefly
                        this.processingOverlay.message.textContent = 'Processing completed! Loading results...';
                        
                        // Reload the page without metadata_only parameter
                        setTimeout(() => {
                            const url = new URL(window.location.href);
                            url.searchParams.delete('metadata_only');
                            window.location.href = url.toString();
                        }, 1500);
                    }
                    // If processing failed, show error
                    else if (data.status === 'failed') {
                        clearInterval(this.processingStatusInterval);
                        this.processingStatusInterval = null;
                        
                        this.processingOverlay.message.textContent = data.message || 'Processing failed';
                        this.processingOverlay.element.classList.add('error');
                    }
                }
            } catch (error) {
                console.error('Error checking processing status:', error);
            }
        }, 5000); // Check every 5 seconds
    }
    
    // Handle subtitle toggling
    toggleSubtitles(enabled) {
        const subtitleDisplay = document.getElementById('subtitle-display');
        if (subtitleDisplay) {
            subtitleDisplay.style.display = enabled ? 'block' : 'none';
        }
    }
    
    getAuthHeaders() {
        const token = localStorage.getItem('access_token');
        const tokenType = localStorage.getItem('token_type');
        
        if (token && tokenType) {
            return {
                'Authorization': `${tokenType} ${token}`
            };
        }
        return {};
    }
}

// Add these helper functions that will be used by the main script
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.hasAttribute('disabled')) return;
            
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to selected tab and content
            tab.classList.add('active');
            const tabId = `${tab.getAttribute('data-tab')}-tab`;
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function initTranscriptControls() {
    // List view
    document.getElementById('transcript-list-view').addEventListener('click', function() {
        document.getElementById('transcript-container').classList.remove('paragraph-view');
        this.classList.add('active');
        document.getElementById('transcript-paragraph-view').classList.remove('active');
    });
    
    // Paragraph view
    document.getElementById('transcript-paragraph-view').addEventListener('click', function() {
        document.getElementById('transcript-container').classList.add('paragraph-view');
        this.classList.add('active');
        document.getElementById('transcript-list-view').classList.remove('active');
    });
    
    // Toggle timestamps
    document.getElementById('transcript-time-toggle').addEventListener('click', function() {
        const container = document.getElementById('transcript-container');
        container.classList.toggle('hide-timestamps');
        this.classList.toggle('active');
    });
    
    // Auto scroll
    document.getElementById('transcript-auto-scroll').addEventListener('click', function() {
        this.classList.toggle('active');
        // The actual scrolling will be handled in the highlightCurrentTranscript function
    });
    
    // Download transcript
    document.getElementById('transcript-download').addEventListener('click', function() {
        const transcriptText = document.querySelectorAll('.transcript-segment')
            .map(segment => {
                const timestamp = segment.querySelector('.transcript-timestamp');
                const text = segment.querySelector('.transcript-text');
                return `${timestamp ? timestamp.textContent : ''} ${text.textContent}`;
            })
            .join('\n\n');
        
        const blob = new Blob([transcriptText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        const videoTitle = document.getElementById('video-title').textContent || 'transcript';
        a.download = `${videoTitle.replace(/[^a-z0-9]/gi, '-').toLowerCase()}-transcript.txt`;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
    
    // Copy transcript
    document.getElementById('transcript-copy').addEventListener('click', function() {
        const transcriptText = Array.from(document.querySelectorAll('.transcript-segment'))
            .map(segment => {
                const timestamp = segment.querySelector('.transcript-timestamp');
                const text = segment.querySelector('.transcript-text');
                return `${timestamp ? timestamp.textContent : ''} ${text.textContent}`;
            })
            .join('\n\n');
        
        navigator.clipboard.writeText(transcriptText).then(() => {
            showToast('Transcript copied to clipboard!');
        });
    });
}

function initShareModal() {
    const modal = document.getElementById('share-modal');
    const shareBtn = document.getElementById('share-button');
    const closeBtn = document.querySelector('.close-modal');
    const copyBtn = document.getElementById('copy-share-link');
    
    // Set up the share link
    const shareLink = window.location.href;
    document.getElementById('share-link-input').value = shareLink;
    
    // Show modal when share button is clicked
    shareBtn.addEventListener('click', () => {
        modal.style.display = 'flex';
    });
    
    // Hide modal when close button is clicked
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    // Hide modal when clicking outside the modal content
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Copy share link button
    copyBtn.addEventListener('click', () => {
        const input = document.getElementById('share-link-input');
        input.select();
        document.execCommand('copy');
        showToast('Link copied to clipboard!');
    });
    
    // Social share buttons
    document.querySelector('.share-btn.twitter').addEventListener('click', () => {
        const text = `Check out this video: ${document.getElementById('video-title').textContent}`;
        const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(shareLink)}`;
        window.open(url, '_blank');
    });
    
    document.querySelector('.share-btn.facebook').addEventListener('click', () => {
        const url = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareLink)}`;
        window.open(url, '_blank');
    });
    
    document.querySelector('.share-btn.linkedin').addEventListener('click', () => {
        const title = document.getElementById('video-title').textContent;
        const url = `https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(shareLink)}&title=${encodeURIComponent(title)}`;
        window.open(url, '_blank');
    });
    
    document.querySelector('.share-btn.email').addEventListener('click', () => {
        const title = document.getElementById('video-title').textContent;
        const body = `Check out this video: ${title}\n${shareLink}`;
        const url = `mailto:?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
        window.location.href = url;
    });
}

function initSubtitleToggle() {
    const subtitleToggle = document.getElementById('subtitle-toggle');
    const subtitleDisplay = document.getElementById('subtitle-display');
    
    subtitleToggle.addEventListener('change', function() {
        if (this.checked) {
            subtitleDisplay.style.display = 'block';
        } else {
            subtitleDisplay.style.display = 'none';
        }
    });
}

function highlightCurrentTranscript(currentTime) {
    const segments = document.querySelectorAll('.transcript-segment');
    let activeSegment = null;
    
    segments.forEach(segment => {
        const startTime = parseFloat(segment.dataset.start || 0);
        const endTime = parseFloat(segment.dataset.end || 0);
        
        if (currentTime >= startTime && currentTime <= endTime) {
            segment.classList.add('active');
            activeSegment = segment;
        } else {
            segment.classList.remove('active');
        }
    });
    
    // Auto-scroll if enabled
    if (activeSegment && document.getElementById('transcript-auto-scroll').classList.contains('active')) {
        const container = document.getElementById('transcript-container');
        container.scrollTop = activeSegment.offsetTop - container.offsetTop - (container.clientHeight / 2);
    }
}

async function loadSummary(videoId, language) {
    try {
        const response = await fetch(`/api/videos/${videoId}/summary?language=${language}`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            throw new Error('Failed to load summary');
        }
        
        const data = await response.json();
        document.getElementById('summary-text').innerHTML = data.summary || 'No summary available.';
    } catch (error) {
        console.error('Error loading summary:', error);
        document.getElementById('summary-text').innerHTML = 'Failed to load summary.';
    }
}

async function loadTranscript(videoId, language) {
    try {
        const response = await fetch(`/api/videos/${videoId}/transcript?language=${language}`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            throw new Error('Failed to load transcript');
        }
        
        const data = await response.json();
        const container = document.getElementById('transcript-container');
        container.innerHTML = '';
        
        if (!data.segments || data.segments.length === 0) {
            container.innerHTML = '<div class="no-transcript">No transcript available for this language.</div>';
            return;
        }
        
        data.segments.forEach(segment => {
            const div = document.createElement('div');
            div.className = 'transcript-segment';
            div.dataset.start = segment.start;
            div.dataset.end = segment.end;
            
            const timestamp = document.createElement('div');
            timestamp.className = 'transcript-timestamp';
            timestamp.textContent = formatTime(segment.start);
            
            const text = document.createElement('div');
            text.className = 'transcript-text';
            text.textContent = segment.text;
            
            div.appendChild(timestamp);
            div.appendChild(text);
            container.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading transcript:', error);
        document.getElementById('transcript-container').innerHTML = '<div class="error-message">Failed to load transcript.</div>';
    }
}

async function loadSubtitles(videoId, language) {
    try {
        const response = await fetch(`/api/videos/${videoId}/subtitles?language=${language}`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            throw new Error('Failed to load subtitles');
        }
        
        const data = await response.json();
        
        // Find or create video player
        let player = window.videoPlayer;
        if (!player) {
            const videoElement = document.getElementById('video-element');
            const container = document.querySelector('.video-player');
            player = new VideoPlayer(videoElement, container);
            window.videoPlayer = player;
        }
        
        // Load subtitles into the player
        player.loadSubtitles(language, data.subtitles || []);
        
        // Update subtitle track for native player support
        const track = document.getElementById('subtitle-track');
        const trackUrl = `/api/videos/${videoId}/subtitles/vtt?language=${language}`;
        track.src = trackUrl;
        track.srclang = language;
        track.label = getLanguageName(language);
        
    } catch (error) {
        console.error('Error loading subtitles:', error);
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function getLanguageName(code) {
    const languages = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'pl': 'Polish',
        'vi': 'Vietnamese',
        'th': 'Thai'
    };
    return languages[code] || code;
}

function updateAuthUI() {
    const authLinks = document.querySelector('.auth-links');
    
    if (localStorage.getItem('access_token')) {
        // User is logged in
        authLinks.innerHTML = `
            <li><a href="/profile.html" class="nav-link">My Videos</a></li>
            <li><button id="logout-btn" class="btn btn-outline btn-sm">Log Out</button></li>
        `;
        
        // Add logout functionality
        document.getElementById('logout-btn').addEventListener('click', () => {
            localStorage.removeItem('access_token');
            localStorage.removeItem('token_type');
            localStorage.removeItem('user_id');
            localStorage.removeItem('username');
            window.location.href = '/';
        });
    } else {
        // User is not logged in
        authLinks.innerHTML = `
            <li><a href="/login.html" class="nav-link">Log In</a></li>
            <li><a href="/signup.html" class="btn btn-primary btn-sm">Sign Up</a></li>
        `;
    }
}

function showToast(message) {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    
    toastMessage.textContent = message;
    toast.classList.add('show');
    
    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showError(message) {
    const main = document.querySelector('main');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    // Insert at the top of main
    main.insertBefore(errorDiv, main.firstChild);
}

function getAuthHeaders() {
    const token = localStorage.getItem('access_token');
    const type = localStorage.getItem('token_type');
    
    if (token && type) {
        return {
            'Authorization': `${type} ${token}`
        };
    }
    return {};
}