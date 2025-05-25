document.addEventListener('DOMContentLoaded', function() {
    // Authentication-related functions
    function isAuthenticated() {
        const token = localStorage.getItem('access_token');
        return token !== null;
    }

    function updateAuthUI() {
        const authLinks = document.querySelector('.auth-links');
        
        if (isAuthenticated()) {
            // User is logged in - show user menu
            const username = localStorage.getItem('username');
            if (authLinks) {
                authLinks.innerHTML = `
                    <li><a href="/profile.html" class="btn btn-outline">
                        <i class="fas fa-video"></i>
                        My Videos
                    </a></li>
                    <li><a href="#" id="logout-link" class="btn btn-primary">
                        <i class="fas fa-user"></i>
                        ${username || 'User'}
                    </a></li>
                `;
                
                // Add event listener for logout
                const logoutLink = document.getElementById('logout-link');
                if (logoutLink) {
                    logoutLink.addEventListener('click', function(e) {
                        e.preventDefault();
                        logout();
                    });
                }
            }
        } else {
            // User is not logged in - show login/register buttons
            if (authLinks) {
                authLinks.innerHTML = `
                    <li><a href="/login.html" class="btn btn-outline login-btn">
                        <span>Login</span>
                        <i class="fas fa-sign-in-alt"></i>
                    </a></li>
                    <li><a href="/signup.html" class="btn btn-primary">Register Now</a></li>
                `;
            }
        }
    }

    function logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_type');
        localStorage.removeItem('username');
        // Clear any upload progress data on logout
        cleanupAllUploads();
        window.location.href = '/login.html';
    }

    // Add authorization headers to fetch requests
    function fetchWithAuth(url, options = {}) {
        const token = localStorage.getItem('access_token');
        const tokenType = localStorage.getItem('token_type');
        
        if (!token) {
            throw new Error('No authentication token found');
        }

        return fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `${tokenType} ${token}`
            }
        });
    }

    // Update UI based on authentication status
    updateAuthUI();

    // Function to clean up all upload progress data
    function cleanupAllUploads() {
        for (let i = localStorage.length - 1; i >= 0; i--) {
            const key = localStorage.key(i);
            if (key && key.startsWith('upload_progress_')) {
                localStorage.removeItem(key);
            }
        }
    }

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

        // Use chunked upload for better performance
        uploadVideoFileChunked(file);
    }

    async function uploadVideoFileChunked(file) {
        const CHUNK_SIZE = 5 * 1024 * 1024; // Simple 5MB chunks for all files
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const uploadId = generateUploadId();
        
        console.log(`Starting upload: ${file.name}, Size: ${(file.size / (1024 * 1024)).toFixed(1)}MB, Chunks: ${totalChunks}`);
        
        // Show simple progress
        showSimpleProgress('Initializing upload...', 0);

        try {
            // Initialize upload
            const initResponse = await fetch('/api/upload/init-chunked', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `${localStorage.getItem('token_type')} ${localStorage.getItem('access_token')}`
                },
                body: JSON.stringify({
                    upload_id: uploadId,
                    filename: file.name,
                    total_size: file.size,
                    total_chunks: totalChunks,
                    languages: 'en',
                    summary_length: 3
                })
            });

            if (!initResponse.ok) {
                throw new Error(`Failed to initialize upload: ${initResponse.status}`);
            }

            // Upload chunks one by one (no parallel uploads to avoid 400 errors)
            let completedChunks = 0;
            
            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const progress = Math.round((chunkIndex / totalChunks) * 85); // Reserve 15% for processing
                showSimpleProgress(`Uploading chunk ${chunkIndex + 1}/${totalChunks}...`, progress);
                
                await uploadSingleChunk(file, chunkIndex, CHUNK_SIZE, uploadId);
                completedChunks++;
                
                // Small delay to prevent overwhelming the server
                if (chunkIndex < totalChunks - 1) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }

            // Finalize upload
            showSimpleProgress('Finalizing upload...', 90);
            
            const finalizeResponse = await fetch('/api/upload/finalize-chunked', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `${localStorage.getItem('token_type')} ${localStorage.getItem('access_token')}`
                },
                body: JSON.stringify({
                    upload_id: uploadId
                })
            });

            if (!finalizeResponse.ok) {
                throw new Error(`Failed to finalize upload: ${finalizeResponse.status}`);
            }

            // Start processing tracking
            showSimpleProgress('Upload complete! Starting processing...', 95);
            
            setTimeout(() => {
                trackSimpleProcessing(uploadId);
            }, 1000);

        } catch (error) {
            console.error('Upload failed:', error);
            showSimpleProgress(`Upload failed: ${error.message}`, 0, true);
        }
    }

    async function uploadSingleChunk(file, chunkIndex, chunkSize, uploadId) {
        const start = chunkIndex * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);

        const formData = new FormData();
        formData.append('chunk', chunk);
        formData.append('chunk_index', chunkIndex.toString());
        formData.append('upload_id', uploadId);

        const response = await fetch('/api/upload/chunk', {
            method: 'POST',
            headers: {
                'Authorization': `${localStorage.getItem('token_type')} ${localStorage.getItem('access_token')}`
            },
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Chunk ${chunkIndex} failed: ${response.status} - ${errorText}`);
        }

        return chunkIndex;
    }

    function showSimpleProgress(message, progress, isError = false) {
        // Update the detailed processing status
        const detailedStatusElement = document.getElementById('detailed-processing-status');
        if (detailedStatusElement) {
            detailedStatusElement.style.display = 'block';
            
            // Update progress bar
            const progressBar = detailedStatusElement.querySelector('.processing-progress-bar');
            if (progressBar) {
                progressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
                progressBar.style.backgroundColor = isError ? '#dc3545' : '#007bff';
            }
            
            // Update message
            const messageElement = detailedStatusElement.querySelector('.processing-message');
            if (messageElement) {
                messageElement.textContent = message;
            }
            
            // Update header
            const headerIcon = detailedStatusElement.querySelector('.processing-status-header i');
            const headerText = detailedStatusElement.querySelector('.processing-status-header h3');
            
            if (isError) {
                if (headerIcon) headerIcon.className = 'fas fa-exclamation-circle';
                if (headerText) headerText.textContent = 'Upload Failed';
            } else if (progress >= 100) {
                if (headerIcon) headerIcon.className = 'fas fa-check-circle';
                if (headerText) headerText.textContent = 'Processing Complete';
            } else {
                if (headerIcon) headerIcon.className = 'fas fa-cog fa-spin';
                if (headerText) headerText.textContent = 'Processing Your Video';
            }
            
            // Update container class
            const container = detailedStatusElement.querySelector('.processing-status-container');
            if (container) {
                container.classList.remove('completed', 'failed');
                if (isError) {
                    container.classList.add('failed');
                } else if (progress >= 100) {
                    container.classList.add('completed');
                }
            }
        }
    }

    function trackSimpleProcessing(uploadId) {
        let startTime = Date.now();
        let checkCount = 0;
        const maxChecks = 120; // 10 minutes max
        
        const checkStatus = async () => {
            try {
                checkCount++;
                
                if (checkCount > maxChecks) {
                    showSimpleProgress('Processing timeout. Please check back later.', 50, true);
                    return;
                }
                
                const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
                
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                
                const data = await response.json();
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const elapsedText = elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed/60)}m ${elapsed%60}s`;
                
                if (data.status === 'completed') {
                    showSimpleProgress(`Processing completed in ${elapsedText}!`, 100);
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${uploadId}`;
                    }, 2000);
                    return;
                } else if (data.status === 'failed' || data.status === 'error') {
                    showSimpleProgress(data.message || 'Processing failed', 0, true);
                    return;
                } else {
                    // Show progress
                    const progress = Math.max(95, data.progress || 95); // Start from 95% after upload
                    const message = data.message || 'Processing your video...';
                    showSimpleProgress(`${message} (${elapsedText})`, progress);
                    
                    // Continue checking
                    setTimeout(checkStatus, 3000);
                }
                
            } catch (error) {
                console.error('Status check error:', error);
                
                // Continue checking unless too many failures
                if (checkCount < maxChecks) {
                    setTimeout(checkStatus, 5000);
                } else {
                    showSimpleProgress('Unable to check processing status', 50, true);
                }
            }
        };
        
        // Start checking
        setTimeout(checkStatus, 2000);
    }

    function generateUploadId() {
        return 'upload_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function trackProcessing(uploadId) {
        let processingStartTime = Date.now();
        let timeoutId = null;
        let maxRetries = 30; // Increased from 15 to 30 retries (about 10+ minutes)
        let retryCount = 0;
        let pollInterval = 3000; // Start with 3 second intervals (faster initial polling)
        
        // Cleanup function to stop polling
        const cleanup = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };
        
        // Store cleanup function globally so it can be called on page unload
        window.currentProcessingCleanup = cleanup;
        
        const checkStatus = async () => {
            try {
                retryCount++;
                
                // Stop polling after max retries to prevent infinite loops
                if (retryCount > maxRetries) {
                    console.log('Max retries reached, stopping status checks');
                    showProcessingStatus({
                        status: 'timeout',
                        progress: 50,
                        message: 'Processing is taking longer than expected. Your video may still be processing in the background. Please check back later.',
                        elapsedTime: formatTime((Date.now() - processingStartTime) / 1000)
                    });
                    cleanup();
                    return;
                }
                
                const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
                
                // Handle network errors or server restarts
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                console.log(`Processing status (attempt ${retryCount}):`, data);
                
                // Calculate elapsed time
                const totalElapsedTime = Math.floor((Date.now() - processingStartTime) / 1000);
                const totalElapsedTimeFormatted = formatTime(totalElapsedTime);
                
                // Use real backend data
                const statusData = {
                    ...data,
                    elapsedTime: totalElapsedTimeFormatted,
                    totalElapsedSeconds: totalElapsedTime
                };
                
                // Update the UI with real status
                showProcessingStatus(statusData);
                
                if (data.status === 'completed') {
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: `Processing completed successfully! Total time: ${totalElapsedTimeFormatted}`,
                        elapsedTime: totalElapsedTimeFormatted,
                    });
                    
                    cleanup();
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${uploadId}`;
                    }, 2500);
                    
                } else if (data.status === 'failed' || data.status === 'error') {
                    showError(data.message || 'Processing failed');
                    cleanup();
                    
                } else {
                    // More gradual increase in poll intervals
                    if (retryCount > 3) {
                        pollInterval = 5000; // 5 seconds after 3 attempts
                    }
                    if (retryCount > 8) {
                        pollInterval = 8000; // 8 seconds after 8 attempts
                    }
                    if (retryCount > 15) {
                        pollInterval = 12000; // 12 seconds after 15 attempts
                    }
                    
                    // Continue checking status with longer intervals
                    timeoutId = setTimeout(checkStatus, pollInterval);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                
                // Handle specific error cases with longer delays
                if (error.message.includes('502') || error.message.includes('503')) {
                    console.log('Backend is restarting, continuing to check...');
                    timeoutId = setTimeout(checkStatus, 15000); // 15 second delay for server restarts
                } else if (retryCount > maxRetries) {
                    showError('Unable to check processing status. Please refresh the page and try again.');
                    cleanup();
                } else {
                    // Continue checking despite errors with longer delay
                    timeoutId = setTimeout(checkStatus, 8000); // Reduced from 10s to 8s
                }
            }
        };

        // Start checking status after a brief delay
        timeoutId = setTimeout(checkStatus, 1000); // Reduced from 2s to 1s
        
        // Return cleanup function so it can be called externally if needed
        return cleanup;
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

    // URL submission - simplified YouTube processing
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
        
        // Show simple progress
        showSimpleProgress('Submitting YouTube URL...', 5);
        
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
            console.log('YouTube processing started:', data);
            
            // Check if this is a cached result - redirect immediately
            if (data.status === 'completed' && data.cached) {
                showSimpleProgress('Found cached result! Redirecting...', 100);
                setTimeout(() => {
                    window.location.href = `/video.html?id=${data.upload_id || data.video_id}`;
                }, 1000);
                return;
            }
            
            // Check if processing started successfully
            if (data.upload_id || data.video_id) {
                const videoId = data.upload_id || data.video_id;
                showSimpleProgress('YouTube video accepted. Processing...', 10);
                
                // Use simple tracking for YouTube as well
                setTimeout(() => {
                    trackSimpleProcessing(videoId);
                }, 1000);
            } else {
                throw new Error('No video ID in response');
            }
        })
        .catch(error => {
            console.error('YouTube processing error:', error);
            showSimpleProgress(`YouTube processing failed: ${error.message}`, 0, true);
        });
    });

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
    
    // Clean up any running processing when page unloads
    window.addEventListener('beforeunload', function() {
        if (window.currentProcessingCleanup) {
            window.currentProcessingCleanup();
        }
    });
});