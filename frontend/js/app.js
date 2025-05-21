document.addEventListener('DOMContentLoaded', function() {
    // Authentication-related functions
    function isAuthenticated() {
        return localStorage.getItem('access_token') !== null;
    }

    function updateAuthUI() {
        const authLinks = document.querySelector('.auth-links');
        const profileLink = document.querySelector('.profile-link');
        
        if (authLinks) {
            if (isAuthenticated()) {
                // User is logged in - show profile link and logout
                authLinks.innerHTML = `
                    <li><a href="/profile.html" class="btn btn-outline">My Videos</a></li>
                    <li><a href="#" id="logout-link" class="btn">Logout</a></li>
                `;
                
                // Add event listener for logout
                document.getElementById('logout-link').addEventListener('click', function(e) {
                    e.preventDefault();
                    logout();
                });
            } 
            // Only update if not authenticated and login button doesn't have the icon
            else {
                const loginBtn = authLinks.querySelector('.login-btn i');
                // Check if login button exists but doesn't have the icon yet
                if (!loginBtn || !loginBtn.classList.contains('fa-sign-in-alt')) {
                    authLinks.innerHTML = `
                        <li><a href="/login.html" class="btn btn-outline login-btn"><span>Login</span><i class="fas fa-sign-in-alt"></i></a></li>
                        <li><a href="/signup.html" class="btn btn-primary">Register Now</a></li>
                    `;
                }
            }
        }
    }

    function logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_type');
        // Redirect to home page
        window.location.href = '/';
    }

    // Add authorization headers to fetch requests
    function fetchWithAuth(url, options = {}) {
        if (isAuthenticated()) {
            const token = localStorage.getItem('access_token');
            const type = localStorage.getItem('token_type');
            
            options.headers = options.headers || {};
            options.headers['Authorization'] = `${type} ${token}`;
        }
        
        return fetch(url, options);
    }

    // Update UI based on authentication status
    updateAuthUI();

    // URL input field cursor effect
    const urlInput = document.querySelector('.url-input input');
    const cursor = document.querySelector('.cursor');
    
    // Show cursor when the input field or its container is clicked
    urlInput.addEventListener('focus', function() {
        cursor.style.display = 'inline-block';
    });
    
    document.querySelector('.url-input').addEventListener('click', function() {
        urlInput.focus();
        cursor.style.display = 'inline-block';
    });
    
    urlInput.addEventListener('blur', function() {
        cursor.style.display = 'none';
    });
    
    // Move cursor when typing
    urlInput.addEventListener('input', function() {
        // No need to move the cursor as it's fixed at the start position
    });
    
    // Drag and drop functionality
    const uploadArea = document.querySelector('.upload-area');
    const uploadContent = document.querySelector('.upload-content');
    
    // Prevent default behavior
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when drag over
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadContent.classList.add('highlight');
    }
    
    function unhighlight() {
        uploadContent.classList.remove('highlight');
    }
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        const file = files[0]; // For now, just handle the first file
        
        if (!file) {
            alert('No file selected.');
            return;
        }
        
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }

        // Check file size (10GB limit)
        const maxSize = 10 * 1024 * 1024 * 1024; // 10GB in bytes
        if (file.size > maxSize) {
            alert('File is too large. Maximum size is 10GB.');
            return;
        }

        // Handle video file upload
        uploadVideoFile(file);
    }

    function uploadVideoFile(file) {
        if (!isAuthenticated()) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('languages', 'en');
        formData.append('summary_length', '3');

        showProcessingStatus({
            status: 'uploading',
            progress: 0,
            message: `Uploading ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`
        });

        // Upload with progress tracking
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload/video', true);
        
        // Add auth header
        const token = localStorage.getItem('access_token');
        const type = localStorage.getItem('token_type');
        xhr.setRequestHeader('Authorization', `${type} ${token}`);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                showProcessingStatus({
                    status: 'uploading',
                    progress: percentComplete,
                    message: `Uploading: ${percentComplete.toFixed(1)}% complete...`
                });
            }
        };

        xhr.onload = function() {
            if (xhr.status === 200 || xhr.status === 202) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    console.log('Upload response:', response);
                    
                    if (response.upload_id) {
                        showProcessingStatus({
                            status: 'processing',
                            progress: 10, 
                            message: 'Upload complete. Starting video processing...'
                        });
                        trackProcessing(response.upload_id);
                    } else {
                        showError('Upload failed: No upload ID in response');
                        hideLoading();
                    }
                } catch (error) {
                    console.error('Error parsing response:', error);
                    showError('Failed to parse server response');
                    hideLoading();
                }
            } else {
                console.error('Upload failed with status:', xhr.status);
                try {
                    const errorResponse = JSON.parse(xhr.responseText);
                    showError(`Upload failed: ${errorResponse.detail || 'Unknown error'}`);
                } catch (e) {
                    showError(`Upload failed with status ${xhr.status}`);
                }
                hideLoading();
            }
        };

        xhr.onerror = function() {
            console.error('Network error during upload');
            showError('Upload failed. Please check your connection and try again.');
            hideLoading();
        };

        xhr.send(formData);
    }

    function trackProcessing(uploadId) {
        const statusElement = document.querySelector('.processing-status');
        if (!statusElement) return;

        const checkStatus = async () => {
            try {
                const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
                const data = await response.json();
                
                showProcessingStatus(data);
                
                if (data.status === 'completed') {
                    window.location.href = `/video.html?id=${uploadId}`;
                } else if (data.status === 'failed') {
                    showError(data.message || 'Processing failed');
                } else {
                    // Continue checking status
                    setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                showError('Failed to check processing status');
            }
        };

        // Start checking status
        checkStatus();
    }

    function showProcessingStatus(data) {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;

        let statusContainer = document.querySelector('.processing-status');
        if (!statusContainer) {
            statusContainer = document.createElement('div');
            statusContainer.className = 'processing-status';
            mainContent.appendChild(statusContainer);
        }

        const statusClass = data.status === 'failed' ? 'error' : 
                           data.status === 'completed' ? 'completed' : 
                           'processing';

        statusContainer.innerHTML = `
            <div class="status-container ${statusClass}">
                <div class="status-content">
                    <div class="processing-step">${getStatusTitle(data.status)}</div>
                    <div class="status-message">${data.message}</div>
                    <div class="process-progress-bar">
                        <div class="process-progress-filled" style="width: ${data.progress}%"></div>
                    </div>
                    <div class="progress-percentage">${Math.round(data.progress)}%</div>
                    ${getEstimatedTime(data)}
                </div>
            </div>
        `;
    }

    function getStatusTitle(status) {
        const titles = {
            'uploading': 'Uploading Video',
            'processing': 'Processing Video',
            'completed': 'Processing Complete',
            'failed': 'Processing Failed',
            'error': 'Error'
        };
        return titles[status] || 'Processing';
    }

    function getEstimatedTime(data) {
        if (data.status === 'completed' || data.status === 'failed') return '';
        
        // Only show estimated time if we have progress
        if (data.progress > 0) {
            // Simple time estimation based on current progress
            const timeElapsed = (Date.now() - data.startTime) / 1000; // in seconds
            const estimatedTotal = (timeElapsed / data.progress) * 100;
            const timeRemaining = Math.max(0, estimatedTotal - timeElapsed);
            
            return `
                <div class="time-remaining">
                    Estimated time remaining: ${formatTime(timeRemaining)}
                </div>
            `;
        }
        return '';
    }

    function formatTime(seconds) {
        if (seconds < 60) return `${Math.round(seconds)} seconds`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')} minutes`;
    }

    function showError(message) {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;

        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <h3>Error</h3>
            <p>${message}</p>
            <button class="btn btn-primary" onclick="location.reload()">Try Again</button>
        `;

        // Replace processing status if it exists
        const statusElement = document.querySelector('.processing-status');
        if (statusElement) {
            statusElement.replaceWith(errorElement);
        } else {
            mainContent.appendChild(errorElement);
        }
    }

    // URL submission - also modified to use authenticated fetch
    const summarizeBtn = document.querySelector('.summarize-btn');
    
    summarizeBtn.addEventListener('click', function() {
        const url = urlInput.value.trim();
        
        if (!url) {
            alert('Please enter a YouTube URL.');
            return;
        }
        
        // Validate if it's a YouTube URL
        if (!isValidYouTubeUrl(url)) {
            alert('Please enter a valid YouTube URL.');
            return;
        }
        
        // Show processing UI
        showProcessingStatus({
            status: 'processing',
            progress: 5,
            message: 'Submitting YouTube URL for processing...',
            startTime: Date.now()
        });
        
        // Send the URL to the backend using authenticated fetch
        fetchWithAuth('/api/youtube/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        })
        .then(response => {
            console.log('YouTube response status:', response.status);
            if (!response.ok) {
                if (response.status === 401) {
                    // Unauthorized - redirect to login
                    window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
                    throw new Error('Please login to process YouTube videos');
                }
                return response.text().then(text => {
                    throw new Error(`Server returned ${response.status}: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Success:', data);
            // Check if processing started successfully
            if (data.video_id) {
                showProcessingStatus({
                    status: 'processing',
                    progress: 10,
                    message: 'YouTube video accepted. Starting extraction...',
                    startTime: Date.now()
                });
                
                // Begin tracking the processing status
                trackYouTubeProcessing(data.video_id);
            } else {
                throw new Error('No video ID in response');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError(`YouTube processing error: ${error.message}`);
            hideLoading();
        });
    });
    
    function trackYouTubeProcessing(videoId) {
        const statusElement = document.querySelector('.processing-status');
        if (!statusElement) return;

        const startTime = Date.now();
        let processingStep = 1;
        const processingSteps = [
            { progress: 10, message: "Extracting video information..." },
            { progress: 20, message: "Downloading audio..." },
            { progress: 40, message: "Transcribing audio..." },
            { progress: 60, message: "Generating summary..." },
            { progress: 80, message: "Preparing results..." }
        ];
        
        const showNextStep = () => {
            if (processingStep < processingSteps.length) {
                const step = processingSteps[processingStep];
                showProcessingStatus({
                    status: 'processing',
                    progress: step.progress,
                    message: step.message,
                    startTime: startTime
                });
                processingStep++;
            }
        };
        
        // Schedule progressive steps to show activity
        const stepInterval = setInterval(() => {
            showNextStep();
        }, 8000); // Show a new step approximately every 8 seconds

        const checkStatus = async () => {
            try {
                const response = await fetchWithAuth(`/api/youtube/status/${videoId}`);
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                const data = await response.json();
                
                console.log('YouTube processing status:', data);
                
                if (data.status === 'completed') {
                    // Clear the step interval
                    clearInterval(stepInterval);
                    
                    // Show completion
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: 'Processing completed successfully!',
                        startTime: startTime
                    });
                    
                    // Wait a moment to show the completion message
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${videoId}`;
                    }, 1500);
                    
                } else if (data.status === 'failed') {
                    // Clear the step interval
                    clearInterval(stepInterval);
                    
                    showError(data.message || 'Processing failed');
                    
                } else {
                    // Continue checking
                    setTimeout(checkStatus, 3000);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                // Still continue checking despite errors
                setTimeout(checkStatus, 5000);
            }
        };

        // Start checking status
        setTimeout(checkStatus, 2000);
        
        // Show first step immediately
        showNextStep();
    }

    function isValidYouTubeUrl(url) {
        // Simple validation for YouTube URLs
        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
        return youtubeRegex.test(url);
    }
    
    // Loading state functions
    function showLoading() {
        // Create or show loading overlay
        let loadingOverlay = document.querySelector('.loading-overlay');
        
        if (!loadingOverlay) {
            loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="spinner"></div>
                <p>Processing your video...</p>
            `;
            document.body.appendChild(loadingOverlay);
        } else {
            loadingOverlay.style.display = 'flex';
        }
    }
    
    function hideLoading() {
        const loadingOverlay = document.querySelector('.loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    // Add click handler for the upload area to trigger file selection
    uploadArea.addEventListener('click', function() {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'video/*';
        fileInput.style.display = 'none';
        
        fileInput.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                handleFiles(this.files);
            }
        });
        
        document.body.appendChild(fileInput);
        fileInput.click();
        
        // Clean up the file input element after selection
        fileInput.addEventListener('input', function() {
            document.body.removeChild(fileInput);
        });
    });
    
    // Responsive adjustments
    function adjustForScreenSize() {
        const windowWidth = window.innerWidth;
        const heroTitle = document.querySelector('.hero-section h2');
        
        if (windowWidth <= 480) {
            heroTitle.innerHTML = 'Drop video or enter URL';
        } else {
            heroTitle.innerHTML = 'Drag and drop your video or enter a URL';
        }
    }
    
    // Run on load and resize
    adjustForScreenSize();
    window.addEventListener('resize', adjustForScreenSize);
});