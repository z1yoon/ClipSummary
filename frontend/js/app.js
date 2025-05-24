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

    // Check for ongoing uploads when page loads
    checkForOngoingUploads();

    // Function to check for ongoing uploads in localStorage
    function checkForOngoingUploads() {
        if (!isAuthenticated()) return;

        // Look for any upload progress in localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith('upload_progress_')) {
                try {
                    const progressData = JSON.parse(localStorage.getItem(key));
                    const uploadId = progressData.uploadId;
                    
                    // Only restore uploads that are not completed
                    if (progressData.status === 'uploading' || progressData.status === 'finalizing' || progressData.status === 'processing') {
                        console.log(`Found ongoing upload: ${uploadId}`, progressData);
                        
                        if (progressData.status === 'uploading') {
                            // Show upload progress
                            const uploadProgress = Math.round((progressData.completedChunks / progressData.totalChunks) * 90);
                            showProcessingStatus({
                                status: 'uploading',
                                progress: uploadProgress,
                                message: `Resuming upload: ${progressData.completedChunks}/${progressData.totalChunks} chunks complete (${uploadProgress}%)`
                            });
                            
                            // Note: We can't actually resume chunk uploads, but we can check the backend status
                            setTimeout(() => checkBackendStatus(uploadId), 2000);
                        } else if (progressData.status === 'processing') {
                            // Resume processing tracking
                            showProcessingStatus({
                                status: 'processing',
                                progress: 15,
                                message: 'Resuming video processing tracking...'
                            });
                            
                            trackProcessing(uploadId);
                        }
                    } else if (progressData.status === 'completed') {
                        // Clean up completed uploads after 24 hours
                        const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
                        if (progressData.startTime < oneDayAgo) {
                            localStorage.removeItem(key);
                        }
                    } else if (progressData.status === 'failed') {
                        // Show failed upload notification
                        showError(`Previous upload failed: ${progressData.filename} - ${progressData.error || 'Unknown error'}`);
                        
                        // Clean up failed uploads after showing the error
                        setTimeout(() => {
                            localStorage.removeItem(key);
                        }, 10000);
                    }
                } catch (error) {
                    console.error('Error parsing upload progress:', error);
                    // Remove corrupted data
                    localStorage.removeItem(key);
                }
            }
        }
    }

    // Function to check backend status for resumed uploads
    async function checkBackendStatus(uploadId) {
        try {
            const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
            if (response.ok) {
                const data = await response.json();
                
                if (data.status === 'completed') {
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: 'Upload and processing completed while you were away!'
                    });
                    
                    // Clean up localStorage
                    localStorage.removeItem(`upload_progress_${uploadId}`);
                    
                    // Offer to redirect to video
                    setTimeout(() => {
                        if (confirm('Your video processing completed! Would you like to view it now?')) {
                            window.location.href = `/video.html?id=${uploadId}`;
                        }
                    }, 2000);
                    
                } else if (data.status === 'processing') {
                    // Resume processing tracking
                    trackProcessing(uploadId);
                    
                } else if (data.status === 'failed') {
                    showError(`Upload failed: ${data.message || 'Unknown error'}`);
                    localStorage.removeItem(`upload_progress_${uploadId}`);
                    
                } else {
                    // Upload might still be in progress on backend
                    showProcessingStatus({
                        status: 'processing',
                        progress: 10,
                        message: 'Upload in progress on server...'
                    });
                    
                    // Track the processing
                    trackProcessing(uploadId);
                }
            } else {
                // Upload might not exist anymore
                console.log(`Upload ${uploadId} not found on server, cleaning up`);
                localStorage.removeItem(`upload_progress_${uploadId}`);
            }
        } catch (error) {
            console.error('Error checking backend status:', error);
            // Continue tracking in case it's a temporary network issue
            trackProcessing(uploadId);
        }
    }

    // Function to clean up old upload progress data
    function cleanupOldUploads() {
        const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
        
        for (let i = localStorage.length - 1; i >= 0; i--) {
            const key = localStorage.key(i);
            if (key && key.startsWith('upload_progress_')) {
                try {
                    const progressData = JSON.parse(localStorage.getItem(key));
                    if (progressData.startTime && progressData.startTime < oneDayAgo) {
                        localStorage.removeItem(key);
                        console.log(`Cleaned up old upload progress: ${key}`);
                    }
                } catch (error) {
                    // Remove corrupted data
                    localStorage.removeItem(key);
                }
            }
        }
    }

    // Clean up old uploads on page load
    cleanupOldUploads();

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
        // Optimize chunk size based on file size for better performance
        let CHUNK_SIZE;
        if (file.size > 5 * 1024 * 1024 * 1024) {        // Files > 5GB
            CHUNK_SIZE = 50 * 1024 * 1024;                // 50MB chunks (much larger!)
        } else if (file.size > 2 * 1024 * 1024 * 1024) { // Files > 2GB  
            CHUNK_SIZE = 25 * 1024 * 1024;                // 25MB chunks
        } else if (file.size > 500 * 1024 * 1024) {      // Files > 500MB
            CHUNK_SIZE = 10 * 1024 * 1024;                // 10MB chunks
        } else {
            CHUNK_SIZE = 5 * 1024 * 1024;                 // 5MB chunks for smaller files
        }
        
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const uploadId = generateUploadId();
        
        console.log(`File size: ${(file.size / (1024 * 1024 * 1024)).toFixed(2)}GB, Chunk size: ${(CHUNK_SIZE / (1024 * 1024))}MB, Total chunks: ${totalChunks}`);
        
        showProcessingStatus({
            status: 'uploading',
            progress: 0,
            message: `Starting chunked upload of ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB) - ${totalChunks} chunks of ${(CHUNK_SIZE / (1024 * 1024))}MB each`
        });

        try {
            // First, initialize the upload session
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

            // Store upload progress in localStorage for persistence
            const uploadProgress = {
                uploadId: uploadId,
                filename: file.name,
                totalSize: file.size,
                totalChunks: totalChunks,
                completedChunks: 0,
                status: 'uploading',
                startTime: Date.now()
            };
            localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));

            // Upload chunks in parallel for maximum speed - INCREASED FROM 3 TO 5!
            const maxConcurrent = 5; // Upload 5 chunks simultaneously for faster uploads
            const uploadPromises = [];
            let completedChunks = 0;

            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += maxConcurrent) {
                const batch = [];
                
                for (let i = 0; i < maxConcurrent && (chunkIndex + i) < totalChunks; i++) {
                    const currentChunk = chunkIndex + i;
                    batch.push(uploadChunk(file, currentChunk, CHUNK_SIZE, uploadId, totalChunks));
                }

                // Wait for this batch to complete
                const batchResults = await Promise.all(batch);
                completedChunks += batchResults.length;

                // Update progress in localStorage and UI
                uploadProgress.completedChunks = completedChunks;
                localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));

                // Update progress
                const progress = Math.round((completedChunks / totalChunks) * 90); // Reserve 10% for finalization
                showProcessingStatus({
                    status: 'uploading',
                    progress: progress,
                    message: `Uploading chunks: ${completedChunks}/${totalChunks} complete (${progress}%) - 5 parallel uploads`
                });
            }

            // Update progress for finalization
            uploadProgress.status = 'finalizing';
            localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));

            // Finalize the upload
            showProcessingStatus({
                status: 'uploading',
                progress: 95,
                message: 'Finalizing upload and starting processing...'
            });

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

            const response = await finalizeResponse.json();
            
            // Update progress - upload complete
            uploadProgress.status = 'processing';
            uploadProgress.completedChunks = totalChunks;
            localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));
            
            showProcessingStatus({
                status: 'processing',
                progress: 100,
                message: 'Upload complete! Starting video processing...'
            });

            // Start tracking processing
            trackProcessing(response.upload_id || uploadId);

        } catch (error) {
            console.error('Chunked upload failed:', error);
            
            // Update progress - upload failed
            const uploadProgress = JSON.parse(localStorage.getItem(`upload_progress_${uploadId}`) || '{}');
            uploadProgress.status = 'failed';
            uploadProgress.error = error.message;
            localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));
            
            showError(`Upload failed: ${error.message}`);
        }
    }

    async function uploadChunk(file, chunkIndex, chunkSize, uploadId, totalChunks) {
        const start = chunkIndex * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);

        const formData = new FormData();
        formData.append('chunk', chunk);
        formData.append('chunk_index', chunkIndex.toString());
        formData.append('upload_id', uploadId);

        // Get fresh auth token for each chunk upload
        const token = localStorage.getItem('access_token');
        const tokenType = localStorage.getItem('token_type');

        if (!token || !tokenType) {
            throw new Error('Authentication required. Please login again.');
        }

        const response = await fetch('/api/upload/chunk', {
            method: 'POST',
            headers: {
                'Authorization': `${tokenType} ${token}`
            },
            body: formData
        });

        if (response.status === 401) {
            throw new Error('Authentication expired. Please login again.');
        }

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to upload chunk ${chunkIndex}: ${response.status} - ${errorText}`);
        }

        return chunkIndex;
    }

    function generateUploadId() {
        return 'upload_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function trackProcessing(uploadId) {
        let processingStartTime = Date.now();
        let timeoutId = null;
        let maxRetries = 15; // Reduced from 40 to 15 retries (about 3 minutes with longer intervals)
        let retryCount = 0;
        let pollInterval = 5000; // Start with 5 second intervals
        
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
                    // Increase poll interval after first few checks to reduce server load
                    if (retryCount > 5) {
                        pollInterval = 10000; // 10 seconds after 5 attempts
                    }
                    if (retryCount > 10) {
                        pollInterval = 15000; // 15 seconds after 10 attempts
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
                    timeoutId = setTimeout(checkStatus, 10000);
                }
            }
        };

        // Start checking status after a brief delay
        timeoutId = setTimeout(checkStatus, 2000);
        
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
            
            // Check if this is a cached result - redirect immediately
            if (data.status === 'completed' && data.cached) {
                showProcessingStatus({
                    status: 'completed',
                    progress: 100,
                    message: 'Found cached result! Redirecting...',
                    elapsedTime: '0 seconds'
                });
                
                // Redirect immediately for cached results
                setTimeout(() => {
                    window.location.href = `/video.html?id=${data.upload_id || data.video_id}`;
                }, 1000);
                return;
            }
            
            // Check if processing started successfully
            if (data.upload_id || data.video_id) {
                const videoId = data.upload_id || data.video_id;
                showProcessingStatus({
                    status: 'processing',
                    progress: 10,
                    message: 'YouTube video accepted. Processing in background...',
                    startTime: Date.now()
                });
                
                // Begin tracking the processing status with simple polling
                trackProcessingSimple(videoId);
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
    
    function trackProcessing(uploadId) {
        let processingStartTime = Date.now();
        let timeoutId = null;
        let maxRetries = 15; // Reduced from 40 to 15 retries (about 3 minutes with longer intervals)
        let retryCount = 0;
        let pollInterval = 5000; // Start with 5 second intervals
        
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
                    // Increase poll interval after first few checks to reduce server load
                    if (retryCount > 5) {
                        pollInterval = 10000; // 10 seconds after 5 attempts
                    }
                    if (retryCount > 10) {
                        pollInterval = 15000; // 15 seconds after 10 attempts
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
                    timeoutId = setTimeout(checkStatus, 10000);
                }
            }
        };

        // Start checking status after a brief delay
        timeoutId = setTimeout(checkStatus, 2000);
        
        // Return cleanup function so it can be called externally if needed
        return cleanup;
    }

    function trackProcessingSimple(uploadId) {
        let checkCount = 0;
        let maxChecks = 8; // Only check 8 times maximum
        let timeoutId = null;
        
        const cleanup = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };
        
        window.currentProcessingCleanup = cleanup;
        
        const checkStatus = async () => {
            try {
                checkCount++;
                
                // Stop after max checks - let user manually refresh
                if (checkCount > maxChecks) {
                    showProcessingStatus({
                        status: 'background',
                        progress: 50,
                        message: 'Processing continues in background. Please check back in a few minutes or refresh the page.',
                        elapsedTime: formatTime(checkCount * 10)
                    });
                    cleanup();
                    return;
                }
                
                const response = await fetchWithAuth(`/api/upload/status/${uploadId}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Status check ${checkCount}:`, data);
                
                if (data.status === 'completed') {
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: 'Processing completed! Redirecting...',
                        elapsedTime: formatTime(checkCount * 10)
                    });
                    
                    cleanup();
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${uploadId}`;
                    }, 2000);
                    
                } else if (data.status === 'failed') {
                    showError(data.message || 'Processing failed');
                    cleanup();
                    
                } else {
                    // Show simple progress message
                    showProcessingStatus({
                        status: 'processing',
                        progress: Math.min(checkCount * 12, 95), // Gradual progress increase
                        message: data.message || 'Processing your video in the background...',
                        elapsedTime: formatTime(checkCount * 10)
                    });
                    
                    // Schedule next check with longer intervals (10 seconds)
                    timeoutId = setTimeout(checkStatus, 10000);
                }
            } catch (error) {
                console.warn('Status check failed:', error);
                
                if (checkCount > maxChecks) {
                    showProcessingStatus({
                        status: 'background',
                        progress: 50,
                        message: 'Processing continues in background. Please refresh the page in a few minutes.',
                        elapsedTime: formatTime(checkCount * 10)
                    });
                    cleanup();
                } else {
                    // Continue with longer delay on error
                    timeoutId = setTimeout(checkStatus, 15000);
                }
            }
        };

        // Start checking after 5 seconds
        timeoutId = setTimeout(checkStatus, 5000);
        return cleanup;
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
    
    // Clean up any running processing when page unloads
    window.addEventListener('beforeunload', function() {
        if (window.currentProcessingCleanup) {
            window.currentProcessingCleanup();
        }
    });
});