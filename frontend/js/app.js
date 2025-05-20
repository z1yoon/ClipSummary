document.addEventListener('DOMContentLoaded', function() {
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
        
        if (file && file.type.startsWith('video/')) {
            // Handle video file upload
            uploadVideoFile(file);
        } else {
            alert('Please upload a valid video file.');
        }
    }
    
    function uploadVideoFile(file) {
        // Create FormData object to send the file
        const formData = new FormData();
        formData.append('video', file);
        
        // Show loading state
        showLoading();
        
        // Send the file to the backend
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Success:', data);
            // Redirect to results page or show results
            if (data.redirectUrl) {
                window.location.href = data.redirectUrl;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error uploading your file. Please try again.');
        })
        .finally(() => {
            hideLoading();
        });
    }
    
    // URL submission
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
        
        // Show loading state
        showLoading();
        
        // Send the URL to the backend
        fetch('/api/youtube', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Success:', data);
            // Redirect to results page or show results
            if (data.redirectUrl) {
                window.location.href = data.redirectUrl;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error processing the YouTube URL. Please try again.');
        })
        .finally(() => {
            hideLoading();
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
});