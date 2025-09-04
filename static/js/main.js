document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('file-info');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const durationEl = document.getElementById('audio-duration');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const progressBar = document.getElementById('progress-bar');
    const muteBtn = document.getElementById('mute-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const removeFileBtn = document.getElementById('remove-file');
    const resultsSection = document.getElementById('results');
    const topPrediction = document.getElementById('top-prediction');
    const otherPredictions = document.getElementById('other-predictions');
    const analyzeAnotherBtn = document.getElementById('analyze-another');
    const mobileMenuBtn = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');

    // Audio variables
    let audioContext;
    let audioBuffer = null;
    let audioSource = null;
    let isPlaying = false;
    let pauseTime = 0;
    let startTime = 0;
    let selectedFile = null;
    let isMuted = false;
    let volume = 0.7;
    let animationId = null;

    // Initialize audio context on user interaction
    function initAudioContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    }
    
    // Mobile menu toggle
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
            mobileMenuBtn.querySelector('i').classList.toggle('fa-bars');
            mobileMenuBtn.querySelector('i').classList.toggle('fa-times');
        });
    }
    
    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('border-green-400', 'bg-opacity-20');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('border-green-400', 'bg-opacity-20');
        });
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (isValidFileType(file)) {
                selectedFile = file;
                updateFileInfo(file);
                loadAudioFile(file);
                fileInfo.classList.remove('hidden');
                dropArea.classList.add('hidden');
                audioPlayerContainer.classList.remove('hidden');
            } else {
                showAlert('Please upload a valid audio file (WAV, MP3, OGG, FLAC, M4A)');
            }
        }
    }
    
    function updateFileInfo(file) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function loadAudioFile(file) {
        if (!file) return;
        initAudioContext();
        
        // Show file info
        updateFileInfo(file);
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const arrayBuffer = e.target.result;
            audioContext.decodeAudioData(arrayBuffer)
                .then(buffer => {
                    audioBuffer = buffer;
                    setupAudioPlayer();
                })
                .catch(error => {
                    console.error('Error decoding audio data', error);
                    showAlert('Error loading audio file. Please try another file.');
                });
        };
        reader.onerror = function() {
            console.error('Error reading file');
            showAlert('Error reading file. Please try again.');
        };
        reader.readAsArrayBuffer(file);
    }
    
    function setupAudioPlayer() {
        if (!audioBuffer) return;
        
        // Show audio player container
        audioPlayerContainer.classList.remove('hidden');
        
        // Update duration display
        const duration = audioBuffer.duration;
        durationEl.textContent = `0:00 / ${formatTime(duration)}`;
        
        // Reset progress bar
        progressBar.style.width = '0%';
        
        // Set up audio source
        if (audioSource) {
            audioSource.stop();
            audioSource = null;
        }
        
        // Set up play/pause button
        if (playPauseBtn) {
            playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            isPlaying = false;
        }
    }
    
    function playAudio() {
        if (!audioBuffer || !progressBar) return;
        
        try {
            // Create audio source
            audioSource = audioContext.createBufferSource();
            audioSource.buffer = audioBuffer;
            
            const gainNode = audioContext.createGain();
            gainNode.gain.value = isMuted ? 0 : volume;

            audioSource.connect(gainNode);
            gainNode.connect(audioContext.destination);

            const offset = pauseTime;
            audioSource.start(0, offset);
            startTime = audioContext.currentTime - offset;

            isPlaying = true;
            if (playPauseBtn) {
                playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
            }

            function updateProgress() {
                if (!isPlaying || !audioBuffer) return;

                const elapsed = audioContext.currentTime - startTime;
                const progress = Math.min(elapsed / audioBuffer.duration, 1);
                progressBar.style.width = `${progress * 100}%`;
                durationEl.textContent = `${formatTime(elapsed)} / ${formatTime(audioBuffer.duration)}`;

                if (progress < 1) {
                    animationId = requestAnimationFrame(updateProgress);
                } else {
                    // Reset when audio finishes
                    isPlaying = false;
                    pauseTime = 0;
                    startTime = 0;
                    playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                    progressBar.style.width = '0%';
                    durationEl.textContent = `0:00 / ${formatTime(audioBuffer.duration)}`;
                }
            }

            updateProgress();
            
            if (audioSource) {
                audioSource.onended = function() {
                    isPlaying = false;
                    if (playPauseBtn) {
                        playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                    }
                    cancelAnimationFrame(animationId);
                };
            }
        } catch (error) {
            console.error('Error playing audio:', error);
            if (playPauseBtn) {
                playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            }
            isPlaying = false;
        }
    }
    
    function pauseAudio() {
        if (audioSource) {
            audioSource.stop();
            pauseTime = audioContext.currentTime - startTime;
            isPlaying = false;
            playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            cancelAnimationFrame(animationId);
        }
    }
    
    function togglePlayPause() {
        if (isPlaying) {
            pauseAudio();
        } else {
            playAudio();
        }
    }
    
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Event listeners for audio controls
    playPauseBtn.addEventListener('click', togglePlayPause);
    
    muteBtn.addEventListener('click', () => {
        isMuted = !isMuted;
        muteBtn.innerHTML = isMuted ? '<i class="fas fa-volume-mute"></i>' : '<i class="fas fa-volume-up"></i>';
        
        if (audioSource) {
            const gainNode = audioContext.createGain();
            gainNode.gain.value = isMuted ? 0 : volume;
            audioSource.disconnect();
            audioSource.connect(gainNode).connect(audioContext.destination);
        }
    });
    
    
    // Handle progress bar seeking
    progressBar.parentElement.addEventListener('click', (e) => {
        if (!audioBuffer) return;
        
        const rect = e.target.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        const seekTime = pos * audioBuffer.duration;
        pauseTime = seekTime;
        
        // Update progress bar
        progressBar.style.width = `${pos * 100}%`;
        durationEl.textContent = `${formatTime(seekTime)} / ${formatTime(audioBuffer.duration)}`;
        
        // If playing, restart playback from new position
        if (isPlaying) {
            audioSource.stop();
            playAudio();
        }
    });

    function isValidFileType(file) {
        const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/x-m4a'];
        return validTypes.includes(file.type) || file.name.match(/\.(wav|mp3|ogg|flac|m4a)$/i);
    }

    // Remove selected file
    removeFileBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        resetAudioPlayer();
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        dropArea.classList.remove('hidden');
        audioPlayerContainer.classList.add('hidden');
    });
    
    function resetAudioPlayer() {
        if (audioSource) {
            audioSource.stop();
            audioSource = null;
        }
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
            audioContext = null;
        }
        isPlaying = false;
        pauseTime = 0;
        startTime = 0;
        cancelAnimationFrame(animationId);
        progressBar.style.width = '0%';
        durationEl.textContent = `0:00 / ${audioBuffer ? formatTime(audioBuffer.duration) : '0:00'}`;
    }

    // Analyze button
    analyzeBtn.addEventListener('click', async function() {
        if (!selectedFile) {
            showAlert('Please select an audio file first');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtnText.classList.add('hidden');
        analyzeSpinner.classList.remove('hidden');
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            // Make API request to the backend
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Display results
            displayResults(result);
            
            // Show results section
            resultsSection.classList.remove('hidden');
            dropArea.parentElement.parentElement.classList.add('hidden');
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error analyzing audio:', error);
            showAlert('Failed to analyze audio. Please try again.');
        } finally {
            // Reset analyze button
            analyzeBtn.disabled = false;
            analyzeBtnText.classList.remove('hidden');
            analyzeSpinner.classList.add('hidden');
        }
    });
    
    // Display prediction results
    function displayResults(predictions) {
        if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
            showAlert('No predictions available for this audio.');
            return;
        }
        
        // Clear previous results
        topPrediction.innerHTML = '';
        otherPredictions.innerHTML = '';
        
        // Display top prediction
        if (predictions.length > 0) {
            const top = predictions[0];
            const topConfidence = Math.round(top.confidence * 100);
            
            const topCard = document.createElement('div');
            topCard.className = 'bg-white rounded-xl shadow-lg overflow-hidden mb-8 transform transition-all duration-300 hover:shadow-xl';
            topCard.innerHTML = `
                <div class="p-6">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <div class="flex items-center">
                                <div class="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center mr-4">
                                    <i class="fas fa-crown text-yellow-500 text-xl"></i>
                                </div>
                                <div>
                                    <h3 class="text-2xl font-bold text-gray-900">${top.species}</h3>
                                    <p class="text-gray-600 italic">${top.scientific_name || 'Scientific name not available'}</p>
                                </div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-3xl font-bold text-green-600">${topConfidence}%</div>
                            <div class="text-sm text-gray-500">Confidence</div>
                        </div>
                    </div>
                    <div class="mt-4">
                        <div class="h-3 bg-gray-200 rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r from-green-400 to-green-600 rounded-full" style="width: ${topConfidence}%"></div>
                        </div>
                    </div>
                </div>
                <div class="bg-gray-50 px-6 py-4 border-t border-gray-100">
                    <div class="flex justify-between items-center">
                        <div class="text-sm text-gray-600">
                            <i class="fas fa-info-circle mr-1"></i> This is the most likely bird species in your recording.
                        </div>
                        <button class="text-green-600 hover:text-green-800 text-sm font-medium">
                            Learn more <i class="fas fa-arrow-right ml-1"></i>
                        </button>
                    </div>
                </div>
            `;
            
            topPrediction.appendChild(topCard);
        }
        
        // Display other predictions
        if (predictions.length > 1) {
            const otherPredictionsContainer = document.createElement('div');
            otherPredictionsContainer.className = 'grid gap-4';
            
            predictions.slice(1).forEach((pred, index) => {
                const confidence = Math.round(pred.confidence * 100);
                const card = document.createElement('div');
                card.className = 'bg-white rounded-lg shadow overflow-hidden hover:shadow-md transition-shadow duration-300';
                card.innerHTML = `
                    <div class="p-4">
                        <div class="flex justify-between items-center">
                            <div class="flex items-center">
                                <div class="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center mr-3 text-gray-600 font-bold">
                                    ${index + 2}
                                </div>
                                <div>
                                    <h4 class="font-semibold text-gray-900">${pred.species}</h4>
                                    <p class="text-sm text-gray-500">${pred.scientific_name || ''}</p>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-lg font-bold text-gray-800">${confidence}%</div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
                                <div class="h-full bg-gradient-to-r from-green-200 to-green-400 rounded-full" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    </div>
                `;
                otherPredictionsContainer.appendChild(card);
            });
            
            otherPredictions.appendChild(otherPredictionsContainer);
        }
    }

    // Analyze another button
    analyzeAnotherBtn.addEventListener('click', function() {
        resetAudioPlayer();
        selectedFile = null;
        fileInput.value = '';
        resultsSection.classList.add('hidden');
        dropArea.parentElement.parentElement.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        dropArea.classList.remove('hidden');
        audioPlayerContainer.classList.add('hidden');
        
        // Reset results
        topPrediction.innerHTML = '';
        otherPredictions.innerHTML = '';
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Show alert message
    function showAlert(message, type = 'error') {
        // Remove any existing alerts
        const existingAlert = document.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }
        
        const alertTypes = {
            error: {
                bg: 'bg-red-50',
                border: 'border-red-400',
                text: 'text-red-700',
                icon: 'exclamation-circle',
                title: 'Error'
            },
            success: {
                bg: 'bg-green-50',
                border: 'border-green-400',
                text: 'text-green-700',
                icon: 'check-circle',
                title: 'Success'
            },
            info: {
                bg: 'bg-blue-50',
                border: 'border-blue-400',
                text: 'text-blue-700',
                icon: 'info-circle',
                title: 'Info'
            },
            warning: {
                bg: 'bg-yellow-50',
                border: 'border-yellow-400',
                text: 'text-yellow-700',
                icon: 'exclamation-triangle',
                title: 'Warning'
            }
        };
        
        const alertStyle = alertTypes[type] || alertTypes.error;
        
        const alert = document.createElement('div');
        alert.className = `alert fixed top-4 right-4 ${alertStyle.bg} ${alertStyle.border} ${alertStyle.text} px-6 py-4 rounded-lg shadow-lg max-w-md z-50 transform transition-all duration-300 translate-x-0 opacity-100`;
        alert.role = 'alert';
        alert.style.transition = 'all 0.3s ease';
        
        alert.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 pt-0.5">
                    <i class="fas fa-${alertStyle.icon} text-lg"></i>
                </div>
                <div class="ml-3">
                    <h3 class="text-sm font-medium">${alertStyle.title}</h3>
                    <div class="mt-1 text-sm">
                        <p>${message}</p>
                    </div>
                </div>
                <div class="ml-4 flex-shrink-0 flex">
                    <button class="inline-flex text-gray-400 hover:text-gray-500 focus:outline-none">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Add close button functionality
        const closeButton = alert.querySelector('button');
        closeButton.addEventListener('click', () => {
            alert.style.transform = 'translateX(100%)';
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.remove();
            }, 300);
        });
        
        document.body.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(alert)) {
                alert.style.transform = 'translateX(100%)';
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.remove();
                }, 300);
            }
        }, 5000);
    }
});
