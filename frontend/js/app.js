document.addEventListener('DOMContentLoaded', function() {
    // Authentication-related functions
    function isAuthenticated() {
        return localStorage.getItem('access_token') !== null;
    }

    function updateAuthUI() {
        const authLinks = document.querySelector('.auth-links');
        
        if (authLinks) {
            if (isAuthenticated()) {
                authLinks.innerHTML = `
                    <li><a href="/profile.html" class="btn btn-outline">My Videos</a></li>
                    <li><a href="#" id="logout-link" class="btn">Logout</a></li>
                `;
                
                document.getElementById('logout-link').addEventListener('click', function(e) {
                    e.preventDefault();
                    logout();
                });
            } else {
                const loginBtn = authLinks.querySelector('.login-btn i');
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
        window.location.href = '/';
    }

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

    function checkForOngoingUploads() {
        if (!isAuthenticated()) return;

        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith('upload_progress_')) {
                try {
                    const progressData = JSON.parse(localStorage.getItem(key));
                    const uploadId = progressData.uploadId;
                    
                    if (progressData.status === 'uploading' || progressData.status === 'processing') {
                        console.log(`Found ongoing upload: ${uploadId}`, progressData);
                        
                        if (progressData.status === 'uploading') {
                            showProcessingStatus({
                                status: 'uploading',
                                progress: 50,
                                message: `Resuming upload check for: ${progressData.filename}`
                            });
                            setTimeout(() => checkBackendStatus(uploadId), 2000);
                        } else if (progressData.status === 'processing') {
                            showProcessingStatus({
                                status: 'processing',
                                progress: 15,
                                message: 'Resuming video processing tracking...'
                            });
                            trackProcessing(uploadId);
                        }
                    } else if (progressData.status === 'completed') {
                        const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
                        if (progressData.startTime < oneDayAgo) {
                            localStorage.removeItem(key);
                        }
                    } else if (progressData.status === 'failed') {
                        showError(`Previous upload failed: ${progressData.filename} - ${progressData.error || 'Unknown error'}`);
                        setTimeout(() => localStorage.removeItem(key), 10000);
                    }
                } catch (error) {
                    console.error('Error parsing upload progress:', error);
                    localStorage.removeItem(key);
                }
            }
        }
    }

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
                    
                    localStorage.removeItem(`upload_progress_${uploadId}`);
                    
                    setTimeout(() => {
                        if (confirm('Your video processing completed! Would you like to view it now?')) {
                            window.location.href = `/video.html?id=${uploadId}`;
                        }
                    }, 2000);
                    
                } else if (data.status === 'processing') {
                    trackProcessing(uploadId);
                } else if (data.status === 'failed') {
                    showError(`Upload failed: ${data.message || 'Unknown error'}`);
                    localStorage.removeItem(`upload_progress_${uploadId}`);
                } else {
                    showProcessingStatus({
                        status: 'processing',
                        progress: 10,
                        message: 'Upload in progress on server...'
                    });
                    trackProcessing(uploadId);
                }
            } else {
                console.log(`Upload ${uploadId} not found on server, cleaning up`);
                localStorage.removeItem(`upload_progress_${uploadId}`);
            }
        } catch (error) {
            console.error('Error checking backend status:', error);
            trackProcessing(uploadId);
        }
    }

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
                    localStorage.removeItem(key);
                }
            }
        }
    }

    cleanupOldUploads();

    // UI Elements
    const urlInput = document.querySelector('.url-input input');
    const cursor = document.querySelector('.cursor');
    const uploadArea = document.querySelector('.upload-area');
    const uploadContent = document.querySelector('.upload-content');
    
    // URL input cursor effect
    if (urlInput && cursor) {
        urlInput.addEventListener('focus', () => cursor.style.display = 'inline-block');
        urlInput.addEventListener('blur', () => cursor.style.display = 'none');
        
        document.querySelector('.url-input').addEventListener('click', function() {
            urlInput.focus();
            cursor.style.display = 'inline-block';
        });
    }
    
    // Drag and drop functionality
    if (uploadArea && uploadContent) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadContent.classList.add('highlight'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadContent.classList.remove('highlight'), false);
        });
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        // Click to upload
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
            
            fileInput.addEventListener('input', () => document.body.removeChild(fileInput));
        });
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        const file = files[0];
        
        if (!file) {
            alert('No file selected.');
            return;
        }
        
        if (!file.type.startsWith('video/')) {
            alert('Please upload a valid video file.');
            return;
        }

        const maxSize = 10 * 1024 * 1024 * 1024; // 10GB
        if (file.size > maxSize) {
            alert('File is too large. Maximum size is 10GB.');
            return;
        }

        uploadVideoFile(file);
    }

    function uploadVideoFile(file) {
        if (!isAuthenticated()) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            return;
        }

        // Use the new secure signed URL approach
        uploadVideoFileSecure(file);
    }

    async function uploadVideoFileSecure(file) {
        try {
            showProcessingStatus({
                status: 'generating_url',
                progress: 5,
                message: `Preparing secure upload for ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`
            });

            // Step 1: Generate signed URL from backend
            const generateUrlResponse = await fetchWithAuth('/api/upload/generate-upload-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: file.name,
                    file_size: file.size,
                    languages: 'en',
                    summary_length: 3
                })
            });

            if (!generateUrlResponse.ok) {
                const errorText = await generateUrlResponse.text();
                throw new Error(`Failed to generate upload URL: ${generateUrlResponse.status} - ${errorText}`);
            }

            const uploadData = await generateUrlResponse.json();
            console.log('Signed URL generated:', { 
                uploadId: uploadData.upload_id,
                expires: uploadData.expires_at 
            });

            // Store upload progress for recovery
            const uploadProgress = {
                uploadId: uploadData.upload_id,
                filename: file.name,
                totalSize: file.size,
                status: 'uploading',
                startTime: Date.now(),
                method: 'signed_url'
            };
            localStorage.setItem(`upload_progress_${uploadData.upload_id}`, JSON.stringify(uploadProgress));

            // Step 2: Upload directly to Azure Blob Storage with progress tracking
            showProcessingStatus({
                status: 'uploading',
                progress: 10,
                message: `Starting upload of ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB) to secure cloud storage...`
            });

            // Use XMLHttpRequest for upload progress tracking
            const uploadResult = await uploadToAzureWithProgress(uploadData, file);
            
            console.log('File uploaded successfully to Azure Blob Storage:', uploadResult);

            // Step 3: Confirm upload completion
            showProcessingStatus({
                status: 'confirming',
                progress: 95,
                message: 'Confirming upload and starting video processing...'
            });

            const confirmResponse = await fetchWithAuth('/api/upload/confirm-upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    upload_id: uploadData.upload_id,
                    filename: file.name
                })
            });

            if (!confirmResponse.ok) {
                const errorText = await confirmResponse.text();
                throw new Error(`Failed to confirm upload: ${confirmResponse.status} - ${errorText}`);
            }

            const confirmData = await confirmResponse.json();
            console.log('Upload confirmed, processing started:', confirmData);

            // Update progress - upload complete, processing started
            uploadProgress.status = 'processing';
            localStorage.setItem(`upload_progress_${uploadData.upload_id}`, JSON.stringify(uploadProgress));

            showProcessingStatus({
                status: 'processing',
                progress: 100,
                message: 'Upload complete! Video processing started...'
            });

            // Start tracking processing
            trackProcessing(uploadData.upload_id);

        } catch (error) {
            console.error('Secure upload failed:', error);
            
            const uploadId = error.uploadId || 'unknown';
            const uploadProgress = JSON.parse(localStorage.getItem(`upload_progress_${uploadId}`) || '{}');
            uploadProgress.status = 'failed';
            uploadProgress.error = error.message;
            localStorage.setItem(`upload_progress_${uploadId}`, JSON.stringify(uploadProgress));
            
            showError(`Upload failed: ${error.message}`);
        }
    }

    function uploadToAzureWithProgress(uploadData, file) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            // Track upload progress
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    const uploadedMB = (event.loaded / (1024 * 1024)).toFixed(2);
                    const totalMB = (event.total / (1024 * 1024)).toFixed(2);
                    const speedMBps = calculateUploadSpeed(event.loaded);
                    
                    // Progress ranges from 10% to 90% during upload
                    const adjustedProgress = 10 + (percentComplete * 0.8); // Maps 0-100% to 10-90%
                    
                    showProcessingStatus({
                        status: 'uploading',
                        progress: adjustedProgress,
                        message: `Uploading ${file.name}: ${uploadedMB}MB / ${totalMB}MB (${percentComplete.toFixed(1)}%) ${speedMBps ? `at ${speedMBps} MB/s` : ''}`
                    });
                }
            });
            
            // Handle upload completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    showProcessingStatus({
                        status: 'uploaded',
                        progress: 90,
                        message: `Upload completed successfully! Verifying file integrity...`
                    });
                    resolve({
                        status: xhr.status,
                        statusText: xhr.statusText,
                        response: xhr.response
                    });
                } else {
                    const errorDetails = {
                        status: xhr.status,
                        statusText: xhr.statusText,
                        response: xhr.responseText,
                        uploadUrl: uploadData.upload_url.split('?')[0] // Log URL without SAS token
                    };
                    console.error('Azure upload failed:', errorDetails);
                    reject(new Error(`Upload to cloud storage failed: ${xhr.status} - ${xhr.statusText || xhr.responseText}`));
                }
            });
            
            // Handle upload errors
            xhr.addEventListener('error', () => {
                console.error('Azure upload network error:', {
                    status: xhr.status,
                    statusText: xhr.statusText,
                    response: xhr.responseText
                });
                reject(new Error('Network error during upload to cloud storage'));
            });
            
            // Handle upload timeout
            xhr.addEventListener('timeout', () => {
                reject(new Error('Upload to cloud storage timed out'));
            });
            
            // Configure the request
            xhr.open(uploadData.upload_method, uploadData.upload_url);
            
            // Set required headers for Azure Blob Storage
            Object.entries(uploadData.headers).forEach(([key, value]) => {
                xhr.setRequestHeader(key, value);
            });
            
            // Set timeout (2 hours for large files)
            xhr.timeout = 2 * 60 * 60 * 1000; // 2 hours in milliseconds
            
            // Start the upload
            xhr.send(file);
        });
    }

    // Upload speed calculation helper
    let uploadStartTime = null;
    let lastUploadedBytes = 0;
    let lastUploadTime = null;

    function calculateUploadSpeed(uploadedBytes) {
        const now = Date.now();
        
        if (!uploadStartTime) {
            uploadStartTime = now;
            lastUploadedBytes = uploadedBytes;
            lastUploadTime = now;
            return null;
        }
        
        // Calculate speed based on recent progress (last 2 seconds)
        const timeDiff = now - lastUploadTime;
        if (timeDiff >= 2000) { // Update speed every 2 seconds
            const bytesDiff = uploadedBytes - lastUploadedBytes;
            const speedBps = bytesDiff / (timeDiff / 1000); // bytes per second
            const speedMBps = (speedBps / (1024 * 1024)).toFixed(1); // MB per second
            
            lastUploadedBytes = uploadedBytes;
            lastUploadTime = now;
            
            return speedMBps;
        }
        
        return null;
    }

    // YouTube URL processing
    const summarizeBtn = document.querySelector('.summarize-btn');
    
    if (summarizeBtn) {
        summarizeBtn.addEventListener('click', function() {
            const url = urlInput.value.trim();
            
            if (!url) {
                alert('Please enter a YouTube URL.');
                return;
            }
            
            if (!isValidYouTubeUrl(url)) {
                alert('Please enter a valid YouTube URL.');
                return;
            }
            
            showProcessingStatus({
                status: 'processing',
                progress: 5,
                message: 'Submitting YouTube URL for processing...',
                startTime: Date.now()
            });
            
            fetchWithAuth('/api/youtube/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            })
            .then(response => {
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
                if (data.status === 'completed' && data.cached) {
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: 'Found cached result! Redirecting...',
                        elapsedTime: '0 seconds'
                    });
                    
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${data.upload_id || data.video_id}`;
                    }, 1000);
                    return;
                }
                
                if (data.upload_id || data.video_id) {
                    const videoId = data.upload_id || data.video_id;
                    showProcessingStatus({
                        status: 'processing',
                        progress: 10,
                        message: 'YouTube video accepted. Processing in background...',
                        startTime: Date.now()
                    });
                    
                    trackProcessingSimple(videoId);
                } else {
                    throw new Error('No video ID in response');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError(`YouTube processing error: ${error.message}`);
            });
        });
    }
    
    function trackProcessing(uploadId) {
        let processingStartTime = Date.now();
        let timeoutId = null;
        let maxRetries = 15;
        let retryCount = 0;
        let pollInterval = 5000;
        
        const cleanup = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
        };
        
        window.currentProcessingCleanup = cleanup;
        
        const checkStatus = async () => {
            try {
                retryCount++;
                
                if (retryCount > maxRetries) {
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
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                const totalElapsedTime = Math.floor((Date.now() - processingStartTime) / 1000);
                const statusData = {
                    ...data,
                    elapsedTime: formatTime(totalElapsedTime),
                    totalElapsedSeconds: totalElapsedTime
                };
                
                showProcessingStatus(statusData);
                
                if (data.status === 'completed') {
                    showProcessingStatus({
                        status: 'completed',
                        progress: 100,
                        message: `Processing completed successfully! Total time: ${formatTime(totalElapsedTime)}`,
                        elapsedTime: formatTime(totalElapsedTime),
                    });
                    
                    cleanup();
                    setTimeout(() => {
                        window.location.href = `/video.html?id=${uploadId}`;
                    }, 2500);
                    
                } else if (data.status === 'failed' || data.status === 'error') {
                    showError(data.message || 'Processing failed');
                    cleanup();
                } else {
                    if (retryCount > 5) pollInterval = 10000;
                    if (retryCount > 10) pollInterval = 15000;
                    
                    timeoutId = setTimeout(checkStatus, pollInterval);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                
                if (error.message.includes('502') || error.message.includes('503')) {
                    timeoutId = setTimeout(checkStatus, 15000);
                } else if (retryCount > maxRetries) {
                    showError('Unable to check processing status. Please refresh the page and try again.');
                    cleanup();
                } else {
                    timeoutId = setTimeout(checkStatus, 10000);
                }
            }
        };

        timeoutId = setTimeout(checkStatus, 2000);
        return cleanup;
    }

    function trackProcessingSimple(uploadId) {
        let checkCount = 0;
        let maxChecks = 8;
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
                    showProcessingStatus({
                        status: 'processing',
                        progress: Math.min(checkCount * 12, 95),
                        message: data.message || 'Processing your video in the background...',
                        elapsedTime: formatTime(checkCount * 10)
                    });
                    
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
                    timeoutId = setTimeout(checkStatus, 15000);
                }
            }
        };

        timeoutId = setTimeout(checkStatus, 5000);
        return cleanup;
    }

    function showProcessingStatus(data) {
        const detailedStatusElement = document.getElementById('detailed-processing-status');
        if (detailedStatusElement) {
            detailedStatusElement.style.display = 'block';
            
            const progressBar = detailedStatusElement.querySelector('.processing-progress-bar');
            if (progressBar) {
                progressBar.style.width = `${Math.round(data.progress)}%`;
            }
            
            const elapsedTimeElement = detailedStatusElement.querySelector('.processing-elapsed');
            if (elapsedTimeElement && data.elapsedTime) {
                elapsedTimeElement.textContent = `Elapsed time: ${data.elapsedTime}`;
            }
            
            const messageElement = detailedStatusElement.querySelector('.processing-message');
            if (messageElement) {
                messageElement.textContent = data.message || 'Processing your video...';
            }
            
            const container = detailedStatusElement.querySelector('.processing-status-container');
            if (container) {
                container.classList.remove('completed', 'failed');
                if (data.status === 'completed') {
                    container.classList.add('completed');
                } else if (data.status === 'failed') {
                    container.classList.add('failed');
                }
            }
            
            const headerIcon = detailedStatusElement.querySelector('.processing-status-header i');
            const headerText = detailedStatusElement.querySelector('.processing-status-header h3');
            
            if (headerIcon && headerText) {
                headerIcon.className = '';
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
            }
        }
    }
    
    function showError(message) {
        const detailedStatusElement = document.getElementById('detailed-processing-status');
        if (detailedStatusElement) {
            detailedStatusElement.style.display = 'block';
            
            const container = detailedStatusElement.querySelector('.processing-status-container');
            if (container) container.classList.add('failed');
            
            const headerIcon = detailedStatusElement.querySelector('.processing-status-header i');
            if (headerIcon) headerIcon.className = 'fas fa-exclamation-circle';
            
            const headerText = detailedStatusElement.querySelector('.processing-status-header h3');
            if (headerText) headerText.textContent = 'Processing Failed';
            
            const messageElement = detailedStatusElement.querySelector('.processing-message');
            if (messageElement) messageElement.textContent = message || 'An error occurred during processing.';
        }
    }

    function formatTime(seconds) {
        if (isNaN(seconds)) return 'Calculating...';
        if (seconds < 60) return `${Math.round(seconds)} seconds`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')} minutes`;
    }

    function isValidYouTubeUrl(url) {
        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
        return youtubeRegex.test(url);
    }
    
    // Responsive adjustments
    function adjustForScreenSize() {
        const windowWidth = window.innerWidth;
        const heroTitle = document.querySelector('.hero-section h2');
        
        if (heroTitle) {
            if (windowWidth <= 480) {
                heroTitle.innerHTML = 'Drop video or enter URL';
            } else {
                heroTitle.innerHTML = 'Drag and drop your video or enter a URL';
            }
        }
    }
    
    adjustForScreenSize();
    window.addEventListener('resize', adjustForScreenSize);

    // Close processing status
    const closeProcessingButton = document.getElementById('close-processing-status');
    if (closeProcessingButton) {
        closeProcessingButton.addEventListener('click', function() {
            const processingStatus = document.getElementById('detailed-processing-status');
            if (processingStatus) {
                processingStatus.style.display = 'none';
            }
        });
    }
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (window.currentProcessingCleanup) {
            window.currentProcessingCleanup();
        }
    });
});