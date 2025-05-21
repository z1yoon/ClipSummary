class VideoPlayer {
    constructor(videoElement, container) {
        this.video = videoElement;
        this.container = container;
        this.currentLanguage = 'en';
        this.availableLanguages = ['en'];
        this.subtitles = {};
        this.currentProgress = 0;
        this.summary = '';
        
        this.initializePlayer();
    }
    
    initializePlayer() {
        // Create video controls
        this.createVideoControls();
        
        // Create subtitle display
        this.createSubtitleDisplay();
        
        // Create content tabs
        this.createContentTabs();
        
        // Add event listeners
        this.addEventListeners();
    }
    
    createVideoControls() {
        const controls = document.createElement('div');
        controls.className = 'video-controls';
        
        // Progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        const progressFilled = document.createElement('div');
        progressFilled.className = 'progress-filled';
        progressBar.appendChild(progressFilled);
        
        // Controls bottom
        const controlsBottom = document.createElement('div');
        controlsBottom.className = 'controls-bottom';
        
        // Left controls
        const leftControls = document.createElement('div');
        leftControls.className = 'left-controls';
        leftControls.innerHTML = `
            <button class="play-pause-btn">
                <i class="fas fa-play"></i>
            </button>
            <div class="volume-container">
                <button class="volume-btn">
                    <i class="fas fa-volume-high"></i>
                </button>
                <div class="volume-slider">
                    <div class="volume-filled"></div>
                </div>
            </div>
            <div class="time-display">0:00 / 0:00</div>
        `;
        
        // Right controls
        const rightControls = document.createElement('div');
        rightControls.className = 'right-controls';
        rightControls.innerHTML = `
            <div class="language-selector">
                <button class="language-btn">
                    <i class="fas fa-closed-captioning"></i>
                    <span>English</span>
                </button>
                <div class="language-dropdown">
                    <div class="dropdown-title">Select Subtitles</div>
                    <div class="language-options"></div>
                    <button class="btn-apply">Apply</button>
                </div>
            </div>
            <button class="fullscreen-btn">
                <i class="fas fa-expand"></i>
            </button>
        `;
        
        controlsBottom.appendChild(leftControls);
        controlsBottom.appendChild(rightControls);
        
        controls.appendChild(progressBar);
        controls.appendChild(controlsBottom);
        
        this.container.appendChild(controls);
        
        // Store elements for later use
        this.controls = {
            progressBar,
            progressFilled,
            playPauseBtn: leftControls.querySelector('.play-pause-btn'),
            volumeBtn: leftControls.querySelector('.volume-btn'),
            volumeSlider: leftControls.querySelector('.volume-slider'),
            volumeFilled: leftControls.querySelector('.volume-filled'),
            timeDisplay: leftControls.querySelector('.time-display'),
            languageBtn: rightControls.querySelector('.language-btn'),
            languageDropdown: rightControls.querySelector('.language-dropdown'),
            languageOptions: rightControls.querySelector('.language-options'),
            applyBtn: rightControls.querySelector('.btn-apply'),
            fullscreenBtn: rightControls.querySelector('.fullscreen-btn')
        };
    }
    
    createSubtitleDisplay() {
        const subtitlesContainer = document.createElement('div');
        subtitlesContainer.className = 'subtitles-container';
        this.container.appendChild(subtitlesContainer);
        this.subtitlesContainer = subtitlesContainer;
    }
    
    createContentTabs() {
        const contentContainer = document.createElement('div');
        contentContainer.className = 'content-container';
        
        // Create tabs
        const tabs = document.createElement('div');
        tabs.className = 'tabs';
        tabs.innerHTML = `
            <button class="tab-btn active" data-tab="summary">Summary</button>
            <button class="tab-btn" data-tab="transcript">Transcript</button>
            <button class="tab-btn" data-tab="settings">Language Settings</button>
        `;
        
        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.className = 'tab-content';
        tabContent.innerHTML = `
            <div id="summary" class="tab-pane active">
                <h2>Summary</h2>
                <div class="summary-content"></div>
            </div>
            <div id="transcript" class="tab-pane">
                <h2>Transcript</h2>
                <div class="transcript-content"></div>
            </div>
            <div id="settings" class="tab-pane">
                <h2>Language Settings</h2>
                <div class="settings-content">
                    <div class="language-settings"></div>
                </div>
            </div>
        `;
        
        contentContainer.appendChild(tabs);
        contentContainer.appendChild(tabContent);
        
        this.container.appendChild(contentContainer);
        
        // Store elements
        this.content = {
            tabs: tabs.querySelectorAll('.tab-btn'),
            panes: tabContent.querySelectorAll('.tab-pane'),
            summary: tabContent.querySelector('.summary-content'),
            transcript: tabContent.querySelector('.transcript-content'),
            settings: tabContent.querySelector('.language-settings')
        };
    }
    
    addEventListeners() {
        // Video controls
        this.video.addEventListener('play', () => this.updatePlayButton());
        this.video.addEventListener('pause', () => this.updatePlayButton());
        this.video.addEventListener('timeupdate', () => this.updateProgress());
        this.video.addEventListener('loadedmetadata', () => this.updateTimeDisplay());
        
        // Control buttons
        this.controls.playPauseBtn.addEventListener('click', () => this.togglePlay());
        this.controls.volumeBtn.addEventListener('click', () => this.toggleMute());
        this.controls.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        
        // Progress bar
        this.controls.progressBar.addEventListener('click', (e) => this.scrub(e));
        
        // Volume slider
        this.controls.volumeSlider.addEventListener('click', (e) => this.updateVolume(e));
        
        // Language selection
        this.controls.languageBtn.addEventListener('click', () => this.toggleLanguageDropdown());
        this.controls.applyBtn.addEventListener('click', () => this.applyLanguageSelection());
        
        // Tabs
        this.content.tabs.forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab));
        });
    }
    
    // Video control methods
    togglePlay() {
        if (this.video.paused) {
            this.video.play();
        } else {
            this.video.pause();
        }
    }
    
    updatePlayButton() {
        const icon = this.controls.playPauseBtn.querySelector('i');
        icon.className = this.video.paused ? 'fas fa-play' : 'fas fa-pause';
    }
    
    updateProgress() {
        const percent = (this.video.currentTime / this.video.duration) * 100;
        this.controls.progressFilled.style.width = `${percent}%`;
        this.updateTimeDisplay();
        this.updateSubtitles();
    }
    
    updateTimeDisplay() {
        const time = `${formatTime(this.video.currentTime)} / ${formatTime(this.video.duration)}`;
        this.controls.timeDisplay.textContent = time;
    }
    
    scrub(e) {
        const rect = this.controls.progressBar.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        this.video.currentTime = percent * this.video.duration;
    }
    
    // Subtitle methods
    loadSubtitles(language, subtitles) {
        this.subtitles[language] = subtitles;
        this.updateLanguageOptions();
    }
    
    updateSubtitles() {
        if (!this.subtitles[this.currentLanguage]) return;
        
        const currentTime = this.video.currentTime;
        const currentSubtitles = this.subtitles[this.currentLanguage].filter(sub => 
            currentTime >= sub.start && currentTime <= sub.end
        );
        
        this.subtitlesContainer.innerHTML = '';
        if (currentSubtitles.length > 0) {
            const subtitle = document.createElement('div');
            subtitle.className = `subtitle ${this.currentLanguage}`;
            subtitle.textContent = currentSubtitles[0].text;
            this.subtitlesContainer.appendChild(subtitle);
        }
        
        // Update transcript highlight
        this.highlightCurrentTranscript(currentTime);
    }
    
    updateLanguageOptions() {
        const options = this.controls.languageOptions;
        options.innerHTML = '';
        
        Object.keys(this.subtitles).forEach(lang => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'subtitle-language';
            input.value = lang;
            input.checked = lang === this.currentLanguage;
            
            label.appendChild(input);
            label.appendChild(document.createTextNode(getLanguageName(lang)));
            options.appendChild(label);
        });
    }
    
    toggleLanguageDropdown() {
        this.controls.languageDropdown.classList.toggle('active');
    }
    
    applyLanguageSelection() {
        const selected = this.controls.languageOptions.querySelector('input:checked');
        if (selected) {
            this.currentLanguage = selected.value;
            this.controls.languageBtn.querySelector('span').textContent = getLanguageName(this.currentLanguage);
            this.updateTranscript();
        }
        this.toggleLanguageDropdown();
    }
    
    // Content methods
    loadSummary(summary) {
        this.summary = summary;
        this.content.summary.innerHTML = `<p>${summary}</p>`;
    }
    
    updateTranscript() {
        const transcript = this.subtitles[this.currentLanguage];
        if (!transcript) return;
        
        const container = this.content.transcript;
        container.innerHTML = '';
        
        transcript.forEach(segment => {
            const entry = document.createElement('div');
            entry.className = 'transcript-entry';
            entry.dataset.start = segment.start;
            entry.dataset.end = segment.end;
            
            const timestamp = document.createElement('span');
            timestamp.className = 'timestamp';
            timestamp.textContent = formatTime(segment.start);
            
            const text = document.createElement('p');
            text.textContent = segment.text;
            
            entry.appendChild(timestamp);
            entry.appendChild(text);
            
            entry.addEventListener('click', () => {
                this.video.currentTime = segment.start;
                if (this.video.paused) {
                    this.video.play();
                }
            });
            
            container.appendChild(entry);
        });
    }
    
    highlightCurrentTranscript(currentTime) {
        const entries = this.content.transcript.querySelectorAll('.transcript-entry');
        entries.forEach(entry => {
            const start = parseFloat(entry.dataset.start);
            const end = parseFloat(entry.dataset.end);
            
            if (currentTime >= start && currentTime <= end) {
                entry.classList.add('active');
                entry.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                entry.classList.remove('active');
            }
        });
    }
    
    switchTab(selectedTab) {
        // Remove active class from all tabs and panes
        this.content.tabs.forEach(tab => tab.classList.remove('active'));
        this.content.panes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to selected tab and pane
        selectedTab.classList.add('active');
        const paneId = selectedTab.dataset.tab;
        document.getElementById(paneId).classList.add('active');
    }
}

// Helper functions
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function getLanguageName(code) {
    const languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi'
    };
    return languages[code] || code;
}