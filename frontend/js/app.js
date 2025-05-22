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
                // Check if login button exists but doesn't have the right icon
                if (!loginBtn || !loginBtn.classList.contains('fa-right-to-bracket')) {
                    authLinks.innerHTML = `
                        <li><a href="/login.html" class="btn btn-outline login-btn"><i class="fa-solid fa-right-to-bracket"></i>Login</a></li>
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
        // Define the expected backend processing stages with their associated messages
        const processingStages = [
            { progress: 15, message: "Loading WhisperX ASR model...", startTime: null },
            { progress: 25, message: "Extracting audio from video...", startTime: null },
            { progress: 40, message: "Transcribing audio with WhisperX...", startTime: null },
            { progress: 60, message: "Aligning transcription with audio...", startTime: null },
            { progress: 75, message: "Generating summary from transcript...", startTime: null },
            { progress: 85, message: "Translating content to requested languages...", startTime: null },
            { progress: 95, message: "Finalizing video processing...", startTime: null }
        ];
        
        let currentStageIndex = 0;
        let lastProgressData = null;
        let processingStartTime = Date.now();
        
        // Initialize the first stage's start time
        processingStages[0].startTime = processingStartTime;
        
        const checkStatus = async () => {
            try {
                const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
                const data = await response.json();
                
                // Keep track of the last data to avoid blinking on network errors
                if (data) {
                    lastProgressData = data;
                }
                
                // Check for specific backend messages to match with our stages
                if (data.message) {
                    const lowerMessage = data.message.toLowerCase();
                    
                    // Check for WhisperX model loading
                    if (lowerMessage.includes('whisperx') && lowerMessage.includes('model')) {
                        if (currentStageIndex < 1) {
                            currentStageIndex = 0;
                            if (!processingStages[0].startTime) {
                                processingStages[0].startTime = Date.now();
                            }
                        }
                    }
                    // Check for audio extraction
                    else if (lowerMessage.includes('extract') && lowerMessage.includes('audio')) {
                        if (currentStageIndex < 2) {
                            currentStageIndex = 1;
                            if (!processingStages[1].startTime) {
                                processingStages[1].startTime = Date.now();
                            }
                        }
                    }
                    // Check for transcription
                    else if (lowerMessage.includes('transcrib')) {
                        if (currentStageIndex < 3) {
                            currentStageIndex = 2;
                            if (!processingStages[2].startTime) {
                                processingStages[2].startTime = Date.now();
                            }
                        }
                    }
                    // Check for alignment
                    else if (lowerMessage.includes('align')) {
                        if (currentStageIndex < 4) {
                            currentStageIndex = 3;
                            if (!processingStages[3].startTime) {
                                processingStages[3].startTime = Date.now();
                            }
                        }
                    }
                    // Check for summarization
                    else if (lowerMessage.includes('summary') || lowerMessage.includes('generat')) {
                        if (currentStageIndex < 5) {
                            currentStageIndex = 4;
                            if (!processingStages[4].startTime) {
                                processingStages[4].startTime = Date.now();
                            }
                        }
                    }
                    // Check for translation
                    else if (lowerMessage.includes('translat')) {
                        if (currentStageIndex < 6) {
                            currentStageIndex = 5;
                            if (!processingStages[5].startTime) {
                                processingStages[5].startTime = Date.now();
                            }
                        }
                    }
                    // Check for finalization
                    else if (lowerMessage.includes('finaliz') || lowerMessage.includes('finish')) {
                        if (currentStageIndex < 7) {
                            currentStageIndex = 6;
                            if (!processingStages[6].startTime) {
                                processingStages[6].startTime = Date.now();
                            }
                        }
                    }
                }
                
                // If backend doesn't provide detailed status, show predicted stages
                if (data.status === 'processing' && data.progress > 0) {
                    // Only move to next stage if we're still processing and progress has increased
                    if (data.progress >= processingStages[currentStageIndex].progress && 
                        currentStageIndex < processingStages.length - 1) {
                        currentStageIndex++;
                        // Set the start time for this stage if it's not set yet
                        if (!processingStages[currentStageIndex].startTime) {
                            processingStages[currentStageIndex].startTime = Date.now();
                        }
                    }
                }
                
                // Calculate the elapsed time for current stage
                const currentStage = processingStages[currentStageIndex];
                const elapsedTimeForStage = currentStage.startTime ? Math.floor((Date.now() - currentStage.startTime) / 1000) : 0;
                const totalElapsedTime = Math.floor((Date.now() - processingStartTime) / 1000);

                // Format times for display
                const elapsedTimeForStageFormatted = formatTime(elapsedTimeForStage);
                const totalElapsedTimeFormatted = formatTime(totalElapsedTime);
                
                // Use backend message if it's specific, otherwise use our staged messages with elapsed time
                let message = data.message;
                if (!message || message.includes('processing') || message.includes('Processing')) {
                    message = `${currentStage.message} (${elapsedTimeForStageFormatted})`;
                }
                
                // Add elapsed time to the data we'll show
                const updatedData = {
                    ...data,
                    message,
                    elapsedTime: totalElapsedTimeFormatted,
                    totalElapsedSeconds: totalElapsedTime,
                    stageElapsedSeconds: elapsedTimeForStage,
                };
                
                // Update the UI with the current status
                showProcessingStatus(updatedData);
                
                if (data.status === 'completed') {
                    const finalElapsedTime = Math.floor((Date.now() - processingStartTime) / 1000);
                    const finalElapsedTimeFormatted = formatTime(finalElapsedTime);
                    
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100, 
                        message: `Processing completed successfully! Total time: ${finalElapsedTimeFormatted}`,
                        elapsedTime: finalElapsedTimeFormatted,
                    });
                    
                    // Wait a moment before redirecting to the results page
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${uploadId}`;
                    }, 2500);
                } else if (data.status === 'failed') {
                    showError(data.message || 'Processing failed');
                } else {
                    // Continue checking status at a reasonable interval
                    setTimeout(checkStatus, 3000);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                
                // If we have previous data, keep showing it to prevent blinking
                if (lastProgressData) {
                    // Move to next stage on error to show progress
                    if (currentStageIndex < processingStages.length - 1) {
                        currentStageIndex++;
                        // Set the start time for this stage if it's not set yet
                        if (!processingStages[currentStageIndex].startTime) {
                            processingStages[currentStageIndex].startTime = Date.now();
                        }
                    }
                    
                    const currentStage = processingStages[currentStageIndex];
                    const elapsedTimeForStage = currentStage.startTime ? Math.floor((Date.now() - currentStage.startTime) / 1000) : 0;
                    const totalElapsedTime = Math.floor((Date.now() - processingStartTime) / 1000);
                    
                    const updatedData = {
                        ...lastProgressData,
                        message: `${currentStage.message} (${formatTime(elapsedTimeForStage)})`,
                        progress: currentStage.progress,
                        elapsedTime: formatTime(totalElapsedTime),
                    };
                    
                    showProcessingStatus(updatedData);
                } else {
                    // Only show error if we have no previous data to display
                    showError('Failed to check processing status');
                }
                
                // Continue checking despite errors
                setTimeout(checkStatus, 5000);
            }
        };

        // Start checking status
        setTimeout(checkStatus, 2000);
    }

    function showProcessingStatus(data) {
        // Show the detailed processing status component
        const detailedStatusElement = document.getElementById('detailed-processing-status');
        detailedStatusElement.style.display = 'block';
        
        // Update progress bar
        const progressBar = detailedStatusElement.querySelector('.processing-progress-bar');
        const progressPercent = Math.round(data.progress);
        progressBar.style.width = `${progressPercent}%`;
        
        // Update elapsed time
        const elapsedTimeElement = detailedStatusElement.querySelector('.processing-elapsed');
        if (data.elapsedTime) {
            elapsedTimeElement.textContent = `Elapsed time: ${data.elapsedTime}`;
        }
        
        // Update status message
        const messageElement = detailedStatusElement.querySelector('.processing-message');
        messageElement.textContent = data.message || 'Processing your video...';
        
        // Update container class based on status
        const container = detailedStatusElement.querySelector('.processing-status-container');
        container.classList.remove('completed', 'failed');
        if (data.status === 'completed') {
            container.classList.add('completed');
        } else if (data.status === 'failed') {
            container.classList.add('failed');
        }
        
        // Update header icon and text based on status
        const headerIcon = detailedStatusElement.querySelector('.processing-status-header i');
        const headerText = detailedStatusElement.querySelector('.processing-status-header h3');
        
        headerIcon.className = ''; // Reset icon classes
        if (data.status === 'completed') {
            headerIcon.className = 'fas fa-check-circle';
            headerText.textContent = 'Processing Complete';
        } else if (data.status === 'failed') {
            headerIcon.className = 'fas fa-exclamation-circle';
            headerText.textContent = 'Processing Failed';
        } else {
            headerIcon.className = 'fas fa-cog fa-spin';
            headerText.textContent = 'Processing Your Video';
        }
        
        // Update stages based on message content
        updateProcessingStages(data);
        
        // Also keep the simple inline status for compatibility
        const inlineStatus = document.querySelector('.inline-processing-status');
        if (inlineStatus) {
            inlineStatus.style.display = 'none'; // Hide the simple status when using detailed view
        }
    }
    
    function updateProcessingStages(data) {
        // Reset all stages first
        const allStages = document.querySelectorAll('.processing-stage');
        allStages.forEach(stage => {
            stage.classList.remove('active', 'completed');
        });
        
        let currentStage = '';
        let currentMessage = (data.message || '').toLowerCase();
        
        // Determine current stage based on message content
        if (currentMessage.includes('upload')) {
            currentStage = 'upload';
        } else if (currentMessage.includes('extract') || currentMessage.includes('audio')) {
            currentStage = 'audio';
        } else if (currentMessage.includes('transcrib') || currentMessage.includes('speech') || 
                  currentMessage.includes('whisper') || currentMessage.includes('convert')) {
            currentStage = 'transcribe';
        } else if (currentMessage.includes('align') || currentMessage.includes('text processing')) {
            currentStage = 'align';
        } else if (currentMessage.includes('speaker') || currentMessage.includes('diariz')) {
            currentStage = 'speaker';
        } else if (currentMessage.includes('summary') || currentMessage.includes('generat')) {
            currentStage = 'summary';
        }
        
        // Calculate which stages are active and completed based on progress
        const progress = data.progress || 0;
        const progressThresholds = {
            'upload': 15,
            'audio': 30,
            'transcribe': 50,
            'align': 70,
            'speaker': 85,
            'summary': 95
        };
        
        // Mark stages as completed or active
        Object.entries(progressThresholds).forEach(([stage, threshold]) => {
            const stageElement = document.querySelector(`.processing-stage[data-stage="${stage}"]`);
            if (!stageElement) return;
            
            // If progress is past this stage's threshold, mark as completed
            if (progress >= threshold) {
                stageElement.classList.add('completed');
                
                // Add elapsed time if available
                if (stage === currentStage && data.stageElapsedSeconds) {
                    const timeElement = stageElement.querySelector('.stage-time');
                    timeElement.textContent = `Time elapsed: ${formatTime(data.stageElapsedSeconds)}`;
                }
            } 
            // If this is the current active stage
            else if (stage === currentStage) {
                stageElement.classList.add('active');
                
                // Add elapsed time if available
                if (data.stageElapsedSeconds) {
                    const timeElement = stageElement.querySelector('.stage-time');
                    timeElement.textContent = `Time elapsed: ${formatTime(data.stageElapsedSeconds)}`;
                }
            }
        });
    }
    
    // Show error with detailed UI
    function showError(message) {
        const detailedStatusElement = document.getElementById('detailed-processing-status');
        if (detailedStatusElement) {
            detailedStatusElement.style.display = 'block';
            
            const container = detailedStatusElement.querySelector('.processing-status-container');
            container.classList.add('failed');
            
            const headerIcon = detailedStatusElement.querySelector('.processing-status-header i');
            headerIcon.className = 'fas fa-exclamation-circle';
            
            const headerText = detailedStatusElement.querySelector('.processing-status-header h3');
            headerText.textContent = 'Processing Failed';
            
            const messageElement = detailedStatusElement.querySelector('.processing-message');
            messageElement.textContent = message || 'An error occurred during processing.';
            
            // Hide the progress stages
            const stages = detailedStatusElement.querySelector('.processing-stages');
            if (stages) {
                stages.style.display = 'none';
            }
        }
        
        // Also show in inline status for compatibility
        const inlineStatus = document.querySelector('.inline-processing-status');
        if (inlineStatus) {
            inlineStatus.style.display = 'flex';
            inlineStatus.style.backgroundColor = 'rgba(220, 53, 69, 0.85)';
            
            const inlineStatusText = inlineStatus.querySelector('.inline-status-text');
            if (inlineStatusText) {
                inlineStatusText.textContent = message || 'Processing failed';
            }
            
            // Reset after a delay
            setTimeout(() => {
                inlineStatus.style.backgroundColor = '';
                inlineStatus.style.display = 'none';
            }, 5000);
        }
    }

    function formatTime(seconds) {
        if (isNaN(seconds)) return 'Calculating...';
        if (seconds < 60) return `${Math.round(seconds)} seconds`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')} minutes`;
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
        // Remove dependency on the statusElement check
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

    // Add event listener for the close button on processing status
    const closeProcessingButton = document.getElementById('close-processing-status');
    if (closeProcessingButton) {
        closeProcessingButton.addEventListener('click', function() {
            const processingStatus = document.getElementById('detailed-processing-status');
            if (processingStatus) {
                processingStatus.style.display = 'none';
            }
        });
    }
});