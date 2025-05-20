// Video player functionality
document.addEventListener('DOMContentLoaded', function() {
    // Video elements
    const videoElement = document.getElementById('video-element');
    const videoContainer = document.querySelector('.video-player');
    const playPauseBtn = document.querySelector('.play-pause-btn');
    const volumeBtn = document.querySelector('.volume-btn');
    const volumeSlider = document.querySelector('.volume-slider');
    const volumeFilled = document.querySelector('.volume-filled');
    const fullscreenBtn = document.querySelector('.fullscreen-btn');
    const progressBar = document.querySelector('.progress-bar');
    const progressFilled = document.querySelector('.progress-filled');
    const timeDisplay = document.querySelector('.time-display');
    const languageBtn = document.querySelector('.language-btn');
    const languageDropdown = document.querySelector('.language-dropdown');
    const applyBtn = document.querySelector('.btn-apply');
    const languageOptions = document.querySelectorAll('.language-options input[type="checkbox"]');
    const subtitlesContainer = document.querySelector('.subtitles-container');
    
    // Tab functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Initialize video source dynamically - would come from server
    function initVideo() {
        // This would typically be populated from backend data
        const videoId = new URLSearchParams(window.location.search).get('id');
        
        if (videoId) {
            // In production, this would be a call to your API to get video details
            console.log(`Loading video with ID: ${videoId}`);
            // For demo purposes, use a sample video
            videoElement.src = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4";
        } else {
            console.log("No video ID provided in URL");
            // Use a placeholder video for demo
            videoElement.src = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4";
        }
    }

    // Helper function to format time
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Update video time display
    function updateTimeDisplay() {
        const currentTime = formatTime(videoElement.currentTime);
        const duration = formatTime(videoElement.duration || 0);
        timeDisplay.textContent = `${currentTime} / ${duration}`;
    }
    
    // Update progress bar
    function updateProgressBar() {
        const percent = (videoElement.currentTime / videoElement.duration) * 100;
        progressFilled.style.width = `${percent}%`;
        updateTimeDisplay();
    }
    
    // Play/pause functionality
    function togglePlay() {
        if (videoElement.paused) {
            videoElement.play();
            playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        } else {
            videoElement.pause();
            playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        }
    }
    
    // Volume functionality
    function toggleMute() {
        videoElement.muted = !videoElement.muted;
        
        if (videoElement.muted) {
            volumeBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
            volumeFilled.style.width = '0%';
        } else {
            volumeBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            volumeFilled.style.width = `${videoElement.volume * 100}%`;
        }
    }
    
    function updateVolume(e) {
        const rect = volumeSlider.getBoundingClientRect();
        const position = (e.clientX - rect.left) / rect.width;
        const volume = Math.max(0, Math.min(1, position));
        
        videoElement.volume = volume;
        volumeFilled.style.width = `${volume * 100}%`;
        
        if (volume === 0) {
            volumeBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>';
            videoElement.muted = true;
        } else {
            volumeBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            videoElement.muted = false;
        }
    }
    
    // Fullscreen functionality
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            if (videoContainer.requestFullscreen) {
                videoContainer.requestFullscreen();
            } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
                videoContainer.webkitRequestFullscreen();
            } else if (videoContainer.msRequestFullscreen) { /* IE11 */
                videoContainer.msRequestFullscreen();
            }
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-compress"></i>';
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) { /* Safari */
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) { /* IE11 */
                document.msExitFullscreen();
            }
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-expand"></i>';
        }
    }
    
    // Progress bar functionality
    function scrub(e) {
        const rect = progressBar.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        videoElement.currentTime = percent * videoElement.duration;
    }
    
    // Subtitle functionality
    let selectedLanguages = ['en']; // Default to English
    
    function updateSubtitles() {
        // This is a simplified example using hardcoded subtitles
        // In a real app, you'd get subtitles from your API
        
        // Clear existing subtitles
        subtitlesContainer.innerHTML = '';
        
        // Current time in seconds
        const currentTime = videoElement.currentTime;
        
        // Example subtitle data (would come from backend)
        const subtitleData = {
            en: [
                { start: 0, end: 3, text: "Hello everyone, welcome to this video." },
                { start: 4, end: 7, text: "Today we'll explore an interesting topic." },
                { start: 8, end: 12, text: "Let's get started." }
            ],
            es: [
                { start: 0, end: 3, text: "Hola a todos, bienvenidos a este video." },
                { start: 4, end: 7, text: "Hoy exploraremos un tema interesante." },
                { start: 8, end: 12, text: "Comencemos." }
            ],
            fr: [
                { start: 0, end: 3, text: "Bonjour à tous, bienvenue dans cette vidéo." },
                { start: 4, end: 7, text: "Aujourd'hui, nous allons explorer un sujet intéressant." },
                { start: 8, end: 12, text: "Commençons." }
            ],
            de: [
                { start: 0, end: 3, text: "Hallo zusammen, willkommen zu diesem Video." },
                { start: 4, end: 7, text: "Heute werden wir ein interessantes Thema erkunden." },
                { start: 8, end: 12, text: "Lass uns anfangen." }
            ],
            zh: [
                { start: 0, end: 3, text: "大家好，欢迎收看此视频。" },
                { start: 4, end: 7, text: "今天我们将探索一个有趣的话题。" },
                { start: 8, end: 12, text: "让我们开始吧。" }
            ],
            ja: [
                { start: 0, end: 3, text: "皆さん、こんにちは。このビデオへようこそ。" },
                { start: 4, end: 7, text: "今日は興味深いトピックを探ります。" },
                { start: 8, end: 12, text: "始めましょう。" }
            ],
            ko: [
                { start: 0, end: 3, text: "여러분, 안녕하세요. 이 비디오에 오신 것을 환영합니다." },
                { start: 4, end: 7, text: "오늘은 흥미로운 주제를 살펴보겠습니다." },
                { start: 8, end: 12, text: "시작하겠습니다." }
            ]
        };
        
        // Display subtitles for each selected language
        selectedLanguages.forEach(lang => {
            if (!subtitleData[lang]) return;
            
            const langSubtitles = subtitleData[lang];
            const currentSubtitles = langSubtitles.filter(sub => 
                currentTime >= sub.start && currentTime <= sub.end
            );
            
            if (currentSubtitles.length > 0) {
                const subtitle = document.createElement('div');
                subtitle.className = `subtitle ${lang}`;
                subtitle.textContent = currentSubtitles[0].text;
                subtitlesContainer.appendChild(subtitle);
            }
        });
    }
    
    // Tab switching functionality
    function switchTab(e) {
        const tabTarget = e.target.dataset.tab;
        
        // Update active state for buttons
        tabBtns.forEach(btn => {
            btn.classList.remove('active');
        });
        e.target.classList.add('active');
        
        // Update active state for content
        tabPanes.forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(tabTarget).classList.add('active');
    }
    
    // Interactive transcript functionality
    function setupTranscriptInteraction() {
        // Make transcript timestamps clickable to jump to that point in the video
        const timestamps = document.querySelectorAll('.timestamp');
        
        timestamps.forEach(timestamp => {
            timestamp.style.cursor = 'pointer';
            timestamp.addEventListener('click', function() {
                const timeString = this.textContent;
                const [minutes, seconds] = timeString.split(':').map(Number);
                const timeInSeconds = minutes * 60 + seconds;
                
                videoElement.currentTime = timeInSeconds;
                if (videoElement.paused) {
                    togglePlay();
                }
            });
        });
    }
    
    // Event listeners
    videoElement.addEventListener('timeupdate', updateProgressBar);
    videoElement.addEventListener('timeupdate', updateSubtitles);
    videoElement.addEventListener('loadedmetadata', updateTimeDisplay);
    videoElement.addEventListener('click', togglePlay);
    
    playPauseBtn.addEventListener('click', togglePlay);
    volumeBtn.addEventListener('click', toggleMute);
    volumeSlider.addEventListener('click', updateVolume);
    fullscreenBtn.addEventListener('click', toggleFullscreen);
    
    let mousedown = false;
    progressBar.addEventListener('click', scrub);
    progressBar.addEventListener('mousedown', () => { mousedown = true; });
    progressBar.addEventListener('mouseup', () => { mousedown = false; });
    progressBar.addEventListener('mousemove', (e) => { if (mousedown) scrub(e); });
    
    // Handle language selection dropdown
    languageBtn.addEventListener('click', function() {
        const isVisible = languageDropdown.style.display === 'block';
        languageDropdown.style.display = isVisible ? 'none' : 'block';
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!languageBtn.contains(e.target) && !languageDropdown.contains(e.target)) {
            languageDropdown.style.display = 'none';
        }
    });
    
    // Apply selected languages
    applyBtn.addEventListener('click', function() {
        selectedLanguages = Array.from(languageOptions)
            .filter(option => option.checked)
            .map(option => option.value);
        
        languageDropdown.style.display = 'none';
        updateSubtitles();
    });
    
    // Tab event listeners
    tabBtns.forEach(btn => {
        btn.addEventListener('click', switchTab);
    });
    
    // Initialize video and features
    initVideo();
    setupTranscriptInteraction();
    
    // Fix for mobile devices
    if ('ontouchstart' in window) {
        videoContainer.addEventListener('touchstart', function() {
            if (videoElement.paused) {
                videoElement.play();
                playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
            } else {
                videoElement.pause();
                playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
            }
        });
    }
});