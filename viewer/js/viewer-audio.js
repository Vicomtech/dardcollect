/**
 * Audio Transcription Viewer Module
 * Handles loading and displaying audio transcriptions with optional audio playback
 */

class AudioTranscriptionViewer {
    constructor() {
        this.items = [];
        this.currentIdx = 0;
        this.mode = 'audio';
        this._handlers = {};
        this.segments = [];  // Current transcription segments
        this._timeUpdateHandler = null;
        this._seekingHandler = null;
        this._seekedHandler = null;
        this._loadAbortController = null;  // Cancel pending loads on switch
    }

    async init(items) {
        ViewerCommon.initializeDomElements();
        
        this.items = items;
        this.currentIdx = 0;
        VIEWER_STATE.imageMode = false;
        
        ViewerCommon.showViewer();
        this.toggleUiVisibility();
        
        this.buildItemList();
        this.attachEventHandlers();
        
        if (items.length > 0) {
            await this.loadItem(0);
        }
    }

    destroy() {
        // Stop audio playback and release resources
        const audioPlayer = document.getElementById('audioPlayer');
        if (audioPlayer) {
            if (this._timeUpdateHandler) {
                audioPlayer.removeEventListener('timeupdate', this._timeUpdateHandler);
            }
            if (this._seekingHandler) {
                audioPlayer.removeEventListener('seeking', this._seekingHandler);
            }
            if (this._seekedHandler) {
                audioPlayer.removeEventListener('seeked', this._seekedHandler);
            }
            audioPlayer.pause();
            audioPlayer.removeAttribute('src');
            audioPlayer.load();  // Release network resources
        }
        // Remove event listeners
        if (this._handlers.documentKeydown) {
            document.removeEventListener('keydown', this._handlers.documentKeydown);
        }
        this.segments = [];
    }

    toggleUiVisibility() {
        // Hide video/image specific UI
        document.getElementById('videoContainer')?.style.setProperty('display', 'none');
        document.getElementById('imageContainer')?.style.setProperty('display', 'none');
        document.getElementById('segmentNav')?.style.setProperty('display', 'none');
        document.getElementById('progressBar')?.style.setProperty('display', 'none');
        
        // Hide video/image buttons
        ['prevFrame', 'nextFrame', 'playPause', 'prevVideo', 'nextVideo', 'prevImage', 'nextImage'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        
        // Show audio-specific controls
        document.getElementById('prevAudio')?.style.setProperty('display', 'inline-block');
        document.getElementById('nextAudio')?.style.setProperty('display', 'inline-block');
        
        // Show audio container
        let audioContainer = document.getElementById('audioContainer');
        if (!audioContainer) {
            // Create audio container if it doesn't exist
            const viewer = document.getElementById('viewer');
            const videoContainer = document.getElementById('videoContainer');
            if (viewer && videoContainer) {
                audioContainer = document.createElement('div');
                audioContainer.id = 'audioContainer';
                audioContainer.style.cssText = 'display: flex; flex-direction: column; gap: 20px; padding: 20px; background: #1a202c; border-radius: 8px; min-height: 300px;';
                audioContainer.innerHTML = `
                    <audio id="audioPlayer" controls preload="auto" style="width: 100%; margin-bottom: 10px; transition: opacity 0.2s;"></audio>
                    <div id="audioTranscription" style="
                        font-size: 1rem;
                        line-height: 1.6;
                        padding: 15px;
                        background: #2d3748;
                        border-radius: 6px;
                        border-left: 4px solid #63b3ed;
                        max-height: 400px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                    "></div>
                `;
                viewer.insertBefore(audioContainer, videoContainer);
            }
        }
        if (audioContainer) {
            audioContainer.style.display = 'flex';
        }
        
        // Show quality metrics area for language info
        document.getElementById('qualityMetrics')?.style.setProperty('display', 'block');
        // Hide video-specific transcription display (we show in audioTranscription instead)
        document.getElementById('transcriptionText')?.style.setProperty('display', 'none');
    }

    buildItemList() {
        const videoList = document.getElementById('videoList');
        if (!videoList) return;
        
        videoList.innerHTML = '';
        this.items.forEach((item, i) => {
            const div = document.createElement('div');
            div.className = 'video-item' + (i === 0 ? ' active' : '');
            // Extract name from transcription path
            const name = item.transcription_path?.split('/').pop()?.replace('.transcription.json', '') || `Audio ${i + 1}`;
            div.textContent = name;
            div.dataset.index = i;
            videoList.appendChild(div);
        });
        
        document.getElementById('videoCount').textContent = this.items.length;
    }

    attachEventHandlers() {
        const self = this;
        const videoList = document.getElementById('videoList');
        
        // Item list click handler
        if (videoList) {
            videoList.addEventListener('click', (e) => {
                const item = e.target.closest('.video-item');
                if (item) {
                    const idx = parseInt(item.dataset.index);
                    if (idx >= 0 && idx < self.items.length) {
                        self.loadItem(idx);
                    }
                }
            });
        }

        // Keyboard handlers
        this._handlers.documentKeydown = (e) => {
            if (e.key === 'n' || e.key === 'ArrowRight') {
                e.preventDefault();
                self.nextItem();
            }
            if (e.key === 'p' || e.key === 'ArrowLeft') {
                e.preventDefault();
                self.prevItem();
            }
        };
        document.addEventListener('keydown', this._handlers.documentKeydown);
    }

    async loadItem(index) {
        if (index < 0 || index >= this.items.length) return;

        // Abort any pending load from previous item
        if (this._loadAbortController) {
            this._loadAbortController.abort();
        }
        this._loadAbortController = new AbortController();
        const signal = this._loadAbortController.signal;

        this.currentIdx = index;
        const item = this.items[index];
        
        // Update list highlighting FIRST (before loading)
        document.querySelectorAll('#videoList .video-item').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });
        
        const audioPlayer = document.getElementById('audioPlayer');
        const transcriptionEl = document.getElementById('audioTranscription');
        const titleEl = document.getElementById('videoTitle');
        const metaEl = document.getElementById('videoMeta');
        const qualityEl = document.getElementById('qualityMetrics');
        
        // Clear previous transcription immediately
        if (transcriptionEl) {
            transcriptionEl.innerHTML = '<em style="color: #a0aec0;">Loading transcription...</em>';
        }
        this.segments = [];
        
        // Update title
        const name = item.transcription_path?.split('/').pop()?.replace('.transcription.json', '') || `Audio ${index + 1}`;
        if (titleEl) titleEl.textContent = `Audio ${index + 1}/${this.items.length}: ${name}`;

        // Load audio with retry logic
        if (audioPlayer && item.audio_path) {
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            audioPlayer.preload = 'auto';
            audioPlayer.style.opacity = '0.5';  // Dim while loading
            audioPlayer.style.display = 'block';
            
            const loadAudioWithRetry = (attempt = 0, maxRetries = 3) => {
                if (signal.aborted) return;  // User switched items
                
                audioPlayer.src = item.audio_path;
                audioPlayer.load();
                
                audioPlayer.oncanplaythrough = () => {
                    audioPlayer.style.opacity = '1';
                };
                
                audioPlayer.onerror = () => {
                    if (signal.aborted) return;
                    
                    if (attempt < maxRetries - 1) {
                        console.warn(`[AUDIO] Attempt ${attempt + 1}/${maxRetries} failed, retrying...`);
                        setTimeout(() => {
                            if (!signal.aborted) {
                                loadAudioWithRetry(attempt + 1, maxRetries);
                            }
                        }, 500 * (attempt + 1));  // Backoff
                    } else {
                        console.error('[AUDIO] All retries failed:', item.audio_path);
                        audioPlayer.style.opacity = '1';  // Remove dim on final failure
                    }
                };
            };
            
            loadAudioWithRetry();
        } else if (audioPlayer) {
            audioPlayer.pause();
            audioPlayer.src = '';
            audioPlayer.style.display = 'none';
        }

        // Load transcription with retry logic
        if (transcriptionEl && item.transcription_path) {
            const maxRetries = 3;
            let lastError = null;
            
            for (let attempt = 0; attempt < maxRetries; attempt++) {
                // Check if user switched to different item
                if (signal.aborted) return;
                
                try {
                    // Use Promise.race for timeout (signal handles user switch cancellation)
                    const fetchPromise = fetch(item.transcription_path, { 
                        signal: signal,
                        cache: 'default'  // Allow caching for faster reload
                    });
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Timeout')), 15000)
                    );
                    
                    const response = await Promise.race([fetchPromise, timeoutPromise]);
                    
                    // Check again after await (user might have switched)
                    if (signal.aborted) return;
                    
                    if (response.ok) {
                        const transData = await response.json();
                        
                        // Final check before updating UI
                        if (signal.aborted) return;
                        
                        this.segments = transData.segments || [];
                        
                        // Display transcription with timestamps if segments available
                        if (this.segments.length > 0) {
                            transcriptionEl.innerHTML = this._renderSegments(this.segments);
                            this._setupTimeSync(audioPlayer);
                        } else if (transData.transcription) {
                            transcriptionEl.textContent = transData.transcription;
                        } else {
                            transcriptionEl.textContent = '(No transcription available)';
                        }
                        
                        // Update meta info
                        if (metaEl) {
                            const parts = [];
                            if (transData.language) parts.push(`Language: ${transData.language}`);
                            if (transData.duration_seconds) parts.push(`${transData.duration_seconds.toFixed(1)}s`);
                            if (transData.transcriber?.model_size) parts.push(`Model: ${transData.transcriber.model_size}`);
                            if (this.segments.length > 0) parts.push(`${this.segments.length} segments`);
                            metaEl.textContent = parts.join(' | ') || '-';
                        }
                        
                        // Update quality/source info
                        if (qualityEl) {
                            qualityEl.innerHTML = '';
                            if (transData.source?.archive_org_id) {
                                const line = document.createElement('div');
                                line.innerHTML = `<strong>Archive.org:</strong> ${transData.source.archive_org_id}`;
                                qualityEl.appendChild(line);
                            }
                            if (transData.parent_audio?.filename) {
                                const line = document.createElement('div');
                                line.innerHTML = `<strong>Source file:</strong> ${transData.parent_audio.filename}`;
                                qualityEl.appendChild(line);
                            }
                            if (transData.transcribed_at) {
                                const line = document.createElement('div');
                                const date = new Date(transData.transcribed_at).toLocaleString();
                                line.innerHTML = `<strong>Transcribed:</strong> ${date}`;
                                qualityEl.appendChild(line);
                            }
                        }
                        break;  // Success - exit retry loop
                    } else {
                        lastError = `HTTP ${response.status}`;
                        if (attempt < maxRetries - 1) {
                            await new Promise(r => setTimeout(r, 500 * (attempt + 1)));  // Backoff
                            continue;
                        }
                        if (!signal.aborted) {
                            transcriptionEl.textContent = `(Failed to load: ${lastError})`;
                        }
                    }
                } catch (err) {
                    // If aborted due to user switching items, exit silently
                    if (err.name === 'AbortError' || signal.aborted) return;
                    
                    lastError = err.message;
                    console.warn(`[AUDIO] Attempt ${attempt + 1}/${maxRetries} failed:`, lastError);
                    
                    if (attempt < maxRetries - 1) {
                        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));  // Backoff
                        continue;
                    }
                    console.error('[AUDIO] All retries failed for transcription:', item.transcription_path);
                    transcriptionEl.textContent = `(Error: ${lastError}. Click to retry)`;
                    transcriptionEl.style.cursor = 'pointer';
                    transcriptionEl.onclick = () => {
                        transcriptionEl.onclick = null;
                        transcriptionEl.style.cursor = '';
                        this.loadItem(index);
                    };
                }
            }
        }
    }

    nextItem() {
        if (this.currentIdx < this.items.length - 1) {
            this.loadItem(this.currentIdx + 1);
        }
    }

    prevItem() {
        if (this.currentIdx > 0) {
            this.loadItem(this.currentIdx - 1);
        }
    }

    /**
     * Format seconds as MM:SS.mmm
     */
    _formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${mins}:${secs.padStart(4, '0')}`;
    }

    /**
     * Render segments with timestamps as clickable spans
     */
    _renderSegments(segments) {
        return segments.map((seg, i) => {
            const start = this._formatTime(seg.start);
            const end = this._formatTime(seg.end);
            return `<div class="transcript-segment" data-index="${i}" data-start="${seg.start}" data-end="${seg.end}" style="
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.2s;
            ">
                <span class="segment-time" style="color: #63b3ed; font-size: 0.85rem; font-family: monospace;">[${start} → ${end}]</span>
                <span class="segment-text" style="margin-left: 8px;">${seg.text}</span>
            </div>`;
        }).join('');
    }

    /**
     * Robust seek with retry logic for unbuffered regions (slow remote connections)
     */
    async _robustSeek(audioPlayer, targetTime, maxRetries = 3) {
        const tolerance = 1.0;  // Accept if within 1 second of target
        
        for (let attempt = 0; attempt < maxRetries; attempt++) {
            // Show seeking state
            audioPlayer.style.opacity = '0.7';
            
            return new Promise((resolve) => {
                let seekTimeout;
                let settled = false;
                
                const cleanup = () => {
                    if (settled) return;
                    settled = true;
                    clearTimeout(seekTimeout);
                    audioPlayer.removeEventListener('seeked', onSeeked);
                    audioPlayer.removeEventListener('error', onError);
                };
                
                const onSeeked = () => {
                    cleanup();
                    const actual = audioPlayer.currentTime;
                    const success = Math.abs(actual - targetTime) < tolerance;
                    
                    if (success) {
                        audioPlayer.style.opacity = '1';
                        resolve(true);
                    } else if (attempt < maxRetries - 1) {
                        // Retry - recursive call
                        console.warn(`[SEEK] Attempt ${attempt + 1}: landed at ${actual.toFixed(1)}s, wanted ${targetTime.toFixed(1)}s. Retrying...`);
                        setTimeout(() => {
                            this._robustSeek(audioPlayer, targetTime, maxRetries - attempt - 1).then(resolve);
                        }, 300);
                    } else {
                        console.error(`[SEEK] Failed to seek to ${targetTime.toFixed(1)}s after ${maxRetries} attempts`);
                        audioPlayer.style.opacity = '1';
                        resolve(false);
                    }
                };
                
                const onError = () => {
                    cleanup();
                    console.warn(`[SEEK] Error on attempt ${attempt + 1}, retrying...`);
                    if (attempt < maxRetries - 1) {
                        setTimeout(() => {
                            this._robustSeek(audioPlayer, targetTime, maxRetries - attempt - 1).then(resolve);
                        }, 500 * (attempt + 1));
                    } else {
                        audioPlayer.style.opacity = '1';
                        resolve(false);
                    }
                };
                
                // Timeout for seek operation (15s for slow connections)
                seekTimeout = setTimeout(() => {
                    cleanup();
                    console.warn(`[SEEK] Timeout on attempt ${attempt + 1}`);
                    if (attempt < maxRetries - 1) {
                        this._robustSeek(audioPlayer, targetTime, maxRetries - attempt - 1).then(resolve);
                    } else {
                        audioPlayer.style.opacity = '1';
                        resolve(false);
                    }
                }, 15000);
                
                audioPlayer.addEventListener('seeked', onSeeked, { once: true });
                audioPlayer.addEventListener('error', onError, { once: true });
                
                // Perform the seek
                try {
                    audioPlayer.currentTime = targetTime;
                } catch (e) {
                    cleanup();
                    resolve(false);
                }
            });
        }
        return false;
    }

    /**
     * Setup time sync: highlight current segment, click to seek
     */
    _setupTimeSync(audioPlayer) {
        const self = this;
        const transcriptionEl = document.getElementById('audioTranscription');
        if (!transcriptionEl || !audioPlayer) return;

        // Remove old handlers
        if (this._timeUpdateHandler) {
            audioPlayer.removeEventListener('timeupdate', this._timeUpdateHandler);
        }
        if (this._seekingHandler) {
            audioPlayer.removeEventListener('seeking', this._seekingHandler);
        }
        if (this._seekedHandler) {
            audioPlayer.removeEventListener('seeked', this._seekedHandler);
        }

        // Track seeks and validate they land correctly (native progress bar clicks)
        let lastSeekTarget = null;
        
        this._seekingHandler = () => {
            audioPlayer.style.opacity = '0.7';
            lastSeekTarget = audioPlayer.currentTime;  // Capture target before seek completes
        };
        audioPlayer.addEventListener('seeking', this._seekingHandler);
        
        // Validate seek completed correctly, retry if it jumped to end
        this._seekedHandler = async () => {
            audioPlayer.style.opacity = '1';
            
            // If we had a target and ended up at the end (>95% of duration), retry
            if (lastSeekTarget !== null && audioPlayer.duration > 0) {
                const actual = audioPlayer.currentTime;
                const nearEnd = actual > audioPlayer.duration * 0.95;
                const targetWasNotEnd = lastSeekTarget < audioPlayer.duration * 0.90;
                const missedByTooMuch = Math.abs(actual - lastSeekTarget) > 2;
                
                if (nearEnd && targetWasNotEnd && missedByTooMuch) {
                    console.warn(`[SEEK] Native seek failed: wanted ${lastSeekTarget.toFixed(1)}s, got ${actual.toFixed(1)}s. Retrying...`);
                    const target = lastSeekTarget;
                    lastSeekTarget = null;  // Prevent infinite loop
                    await self._robustSeek(audioPlayer, target);
                }
            }
            lastSeekTarget = null;
        };
        audioPlayer.addEventListener('seeked', this._seekedHandler);

        // Click to seek (transcription segments)
        transcriptionEl.addEventListener('click', async (e) => {
            const segEl = e.target.closest('.transcript-segment');
            if (segEl && audioPlayer.src) {
                const start = parseFloat(segEl.dataset.start);
                
                // Use robust seek for unbuffered regions
                const success = await self._robustSeek(audioPlayer, start);
                if (success) {
                    audioPlayer.play();
                }
            }
        });

        // Highlight active segment on playback
        let lastActiveIndex = -1;  // Track to scroll only on segment change
        
        this._timeUpdateHandler = () => {
            const currentTime = audioPlayer.currentTime;
            const segments = transcriptionEl.querySelectorAll('.transcript-segment');
            let currentActiveIndex = -1;
            
            segments.forEach((segEl, idx) => {
                const start = parseFloat(segEl.dataset.start);
                const end = parseFloat(segEl.dataset.end);
                const isActive = currentTime >= start && currentTime < end;
                
                segEl.style.background = isActive ? 'rgba(99, 179, 237, 0.2)' : 'transparent';
                segEl.style.borderLeft = isActive ? '3px solid #63b3ed' : '3px solid transparent';
                
                if (isActive) {
                    currentActiveIndex = idx;
                }
            });
            
            // Only scroll when segment changes
            if (currentActiveIndex !== -1 && currentActiveIndex !== lastActiveIndex) {
                lastActiveIndex = currentActiveIndex;
                segments[currentActiveIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        };
        audioPlayer.addEventListener('timeupdate', this._timeUpdateHandler);
    }
}
