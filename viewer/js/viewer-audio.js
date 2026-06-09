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
            audioPlayer.pause();
            audioPlayer.removeAttribute('src');
            audioPlayer.load();  // Release network resources
        }
        // Remove event listeners
        if (this._handlers.documentKeydown) {
            document.removeEventListener('keydown', this._handlers.documentKeydown);
        }
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
                    <audio id="audioPlayer" controls preload="none" style="width: 100%; margin-bottom: 10px;"></audio>
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
        
        // Update title
        const name = item.transcription_path?.split('/').pop()?.replace('.transcription.json', '') || `Audio ${index + 1}`;
        if (titleEl) titleEl.textContent = `Audio ${index + 1}/${this.items.length}: ${name}`;

        // Load audio if available
        if (audioPlayer && item.audio_path) {
            // Stop any current playback first
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            audioPlayer.preload = 'none';  // Prevent aggressive buffering over network
            audioPlayer.src = item.audio_path;
            audioPlayer.load();  // Reset player state
            audioPlayer.style.display = 'block';
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
                    
                    // Display transcription
                    if (transData.transcription) {
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
}
