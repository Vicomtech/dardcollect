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

        this.currentIdx = index;
        const item = this.items[index];
        
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

        // Load audio if available (don't await - let it load in parallel)
        if (audioPlayer && item.audio_path) {
            // Stop any current playback first
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            audioPlayer.preload = 'auto';  // Preload enough for smooth playback
            audioPlayer.style.opacity = '0.5';  // Dim while loading
            audioPlayer.src = item.audio_path;
            audioPlayer.load();  // Reset player state
            audioPlayer.style.display = 'block';
            
            // Restore opacity when ready to play
            audioPlayer.oncanplaythrough = () => {
                audioPlayer.style.opacity = '1';
            };
        } else if (audioPlayer) {
            audioPlayer.pause();
            audioPlayer.src = '';
            audioPlayer.style.display = 'none';
        }

        // Load transcription
        if (transcriptionEl && item.transcription_path) {
            try {
                const response = await fetch(item.transcription_path, { cache: 'no-store' });
                if (response.ok) {
                    const transData = await response.json();
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
                } else {
                    transcriptionEl.textContent = '(Failed to load transcription)';
                }
            } catch (err) {
                console.error('[AUDIO] Error loading transcription:', err);
                transcriptionEl.textContent = '(Error loading transcription)';
            }
        }

        // Update list highlighting
        document.querySelectorAll('#videoList .video-item').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });
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
     * Setup time sync: highlight current segment, click to seek
     */
    _setupTimeSync(audioPlayer) {
        const self = this;
        const transcriptionEl = document.getElementById('audioTranscription');
        if (!transcriptionEl || !audioPlayer) return;

        // Remove old handler
        if (this._timeUpdateHandler) {
            audioPlayer.removeEventListener('timeupdate', this._timeUpdateHandler);
        }

        // Click to seek
        transcriptionEl.addEventListener('click', (e) => {
            const segEl = e.target.closest('.transcript-segment');
            if (segEl && audioPlayer.src) {
                const start = parseFloat(segEl.dataset.start);
                audioPlayer.currentTime = start;
                audioPlayer.play();
            }
        });

        // Highlight active segment on playback
        this._timeUpdateHandler = () => {
            const currentTime = audioPlayer.currentTime;
            const segments = transcriptionEl.querySelectorAll('.transcript-segment');
            
            segments.forEach(segEl => {
                const start = parseFloat(segEl.dataset.start);
                const end = parseFloat(segEl.dataset.end);
                const isActive = currentTime >= start && currentTime < end;
                
                segEl.style.background = isActive ? 'rgba(99, 179, 237, 0.2)' : 'transparent';
                segEl.style.borderLeft = isActive ? '3px solid #63b3ed' : '3px solid transparent';
                
                // Auto-scroll to active segment
                if (isActive) {
                    segEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            });
        };
        audioPlayer.addEventListener('timeupdate', this._timeUpdateHandler);
    }
}
