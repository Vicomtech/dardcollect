/**
 * Video Viewer Module
 * Handles loading, displaying, and navigating video clips with detections
 */

class VideoViewer {
    constructor() {
        this.detections = [];
        this.videoFiles = {};
        this.currentVideoIndex = 0;
        this.currentSegmentIndex = 0;
        this.fps = 30;
        this.mode = 'video';
        this.isSeeking = false;
        this.pendingDelta = 0;
        this.loopGen = 0;
        this.isPlaying = false;
        this.animationFrameId = null;
        this.lastDrawnFrame = -1;  // Track last drawn frame to avoid redundant draws
        this._handlers = {};
        this.transcriptionSegments = [];  // Store transcription segments for subtitles
    }

    async init(items) {
        ViewerCommon.initializeDomElements();
        
        // Mark all items as clips (person_clips folder contains video clips)
        this.detections = items.map(item => {
            item.isClip = true;
            return item;
        });
        
        VIEWER_STATE.videoDetections = items;
        VIEWER_STATE.videoFiles = {};
        VIEWER_STATE.imageMode = false;
        
        ViewerCommon.showViewer();
        ViewerCommon.toggleUiVisibility('video');
        
        this.buildVideoList();
        this.attachEventHandlers();
        
        if (items.length > 0) {
            this.loadVideo(0);
        }
    }

    destroy() {
        // Cancel any pending animation frame
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        // Pause video to stop audio and network buffering
        const videoPlayer = document.getElementById('videoPlayer');
        if (videoPlayer) {
            videoPlayer.pause();
            videoPlayer.src = '';
        }
        // Increment loopGen so any lingering render frame callbacks exit early
        this.loopGen++;
    }

    buildVideoList() {
        const videoList = document.getElementById('videoList');
        if (!videoList) return;
        
        videoList.innerHTML = '';
        this.detections.forEach((item, i) => {
            const div = document.createElement('div');
            div.className = 'video-item' + (i === 0 ? ' active' : '');
            const name = item._fileName || item.json_path?.split('/').pop()?.replace('.json', '') || `Video ${i}`;
            div.textContent = name;
            div.dataset.index = i;
            videoList.appendChild(div);
        });
        
        document.getElementById('videoCount').textContent = this.detections.length;
    }

    attachEventHandlers() {
        const self = this;
        const videoList = document.getElementById('videoList');
        const videoPlayer = document.getElementById('videoPlayer');
        const canvas = document.getElementById('overlayCanvas');
        
        // Video list click handler
        if (videoList) {
            videoList.addEventListener('click', (e) => {
                const item = e.target.closest('.video-item');
                if (item) {
                    const idx = parseInt(item.dataset.index);
                    if (idx >= 0 && idx < self.detections.length) {
                        self.loadVideo(idx);
                    }
                }
            });
        }

        // Button handlers
        document.getElementById('prevFrame')?.addEventListener('click', () => self.stepFrame(-1));
        document.getElementById('nextFrame')?.addEventListener('click', () => self.stepFrame(1));
        document.getElementById('playPause')?.addEventListener('click', () => {
            if (videoPlayer?.paused) {
                videoPlayer?.play();
            } else {
                videoPlayer?.pause();
            }
        });
        document.getElementById('prevVideo')?.addEventListener('click', () => self.prevVideo());
        document.getElementById('nextVideo')?.addEventListener('click', () => self.nextVideo());

        // Progress slider
        const segmentProgress = document.getElementById('segmentProgress');
        if (segmentProgress) {
            segmentProgress.addEventListener('input', () => self.onProgressChange());
        }

        // Keyboard handlers
        document.addEventListener('keydown', (e) => {
            if (!VIEWER_STATE.imageMode) {
                if (e.key === 'ArrowLeft') { e.preventDefault(); self.stepFrame(-1); }
                if (e.key === 'ArrowRight') { e.preventDefault(); self.stepFrame(1); }
                if (e.key === ' ') { 
                    e.preventDefault(); 
                    videoPlayer?.paused ? videoPlayer?.play() : videoPlayer?.pause();
                }
                if (e.key === 'n') self.nextVideo();
                if (e.key === 'p') self.prevVideo();
            }
        });

        // Video events
        if (videoPlayer) {
            videoPlayer.addEventListener('play', () => {
                const btn = document.getElementById('playPause');
                if (btn) btn.textContent = '⏸ Pause';
                self.isPlaying = true;
                self.startRenderLoop();
            });
            
            videoPlayer.addEventListener('pause', () => {
                const btn = document.getElementById('playPause');
                if (btn) btn.textContent = '▶ Play';
                self.isPlaying = false;
                if (self.animationFrameId) {
                    cancelAnimationFrame(self.animationFrameId);
                    self.animationFrameId = null;
                }
            });
            
            videoPlayer.addEventListener('seeked', () => {
                self.isSeeking = false;
                if (self.pendingDelta !== 0) {
                    const delta = self.pendingDelta;
                    self.pendingDelta = 0;
                    self.doSeek(delta);
                } else {
                    self.updateSegmentInfo(null, true);
                }
            });

            // Set container aspect ratio to match video, then sync canvas backing store
            videoPlayer.addEventListener('loadedmetadata', () => {
                const canvas = document.getElementById('overlayCanvas');
                const container = document.getElementById('videoContainer');
                if (!canvas || !container || !videoPlayer.videoWidth || !videoPlayer.videoHeight) return;

                const vw = videoPlayer.videoWidth;
                const vh = videoPlayer.videoHeight;
                const vAspect = vw / vh;
                const maxH = window.innerHeight * 0.65;
                const parentW = container.parentElement?.clientWidth || window.innerWidth;

                // Compute the exact rendered size the container should have
                let renderedH = parentW / vAspect;
                let renderedW = parentW;
                if (renderedH > maxH) {
                    renderedH = maxH;
                    renderedW = renderedH * vAspect;
                }

                // Apply exact dimensions so video and canvas are pixel-perfect aligned
                container.style.width = `${renderedW}px`;
                container.style.height = `${renderedH}px`;
                container.style.aspectRatio = `${vw} / ${vh}`;

                canvas.width = Math.round(renderedW);
                canvas.height = Math.round(renderedH);
                canvas.videoWidth = vw;
                canvas.videoHeight = vh;

                console.log('[VIDEO] loadedmetadata:', vw, 'x', vh);
                console.log('[CONTAINER] rendered:', renderedW.toFixed(2), 'x', renderedH.toFixed(2));
                console.log('[CANVAS] backing store:', canvas.width, 'x', canvas.height);
                console.log('[SCALE] x:', (canvas.width / vw).toFixed(4), 'y:', (canvas.height / vh).toFixed(4));

                // Redraw with confirmed canvas dimensions
                self.lastDrawnFrame = -1;
                self.updateSegmentInfo(null, true);
            });
        }

        // Resize handler: recalculate container and canvas to keep pixel-perfect alignment
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                const canvas = document.getElementById('overlayCanvas');
                const container = document.getElementById('videoContainer');
                const videoPlayer = document.getElementById('videoPlayer');
                if (!canvas || !container || !videoPlayer || !videoPlayer.videoWidth || !container.style.aspectRatio) return;

                const vw = videoPlayer.videoWidth;
                const vh = videoPlayer.videoHeight;
                const vAspect = vw / vh;
                const maxH = window.innerHeight * 0.65;
                const parentW = container.parentElement?.clientWidth || window.innerWidth;

                let renderedH = parentW / vAspect;
                let renderedW = parentW;
                if (renderedH > maxH) {
                    renderedH = maxH;
                    renderedW = renderedH * vAspect;
                }

                const newW = Math.round(renderedW);
                const newH = Math.round(renderedH);
                if (newW !== canvas.width || newH !== canvas.height) {
                    container.style.width = `${renderedW}px`;
                    container.style.height = `${renderedH}px`;
                    canvas.width = newW;
                    canvas.height = newH;
                    self.lastDrawnFrame = -1;
                    self.updateSegmentInfo(null, true);
                    console.log(`[CANVAS] Resized to ${canvas.width}x${canvas.height}`);
                }
            }, 150);
        });

        // Viz control changes - use 'click' for checkboxes
        const checkboxes = [
            document.getElementById('showBBox'),
            document.getElementById('showKeypoints'),
            document.getElementById('showScores'),
            document.getElementById('showIds'),
            document.getElementById('showFaceCropArcface'),
            document.getElementById('showFaceCropOfiq'),
            document.getElementById('showSubtitles')
        ];
        checkboxes.forEach(ctrl => {
            ctrl?.addEventListener('click', () => {
                if (!VIEWER_STATE.imageMode) {
                    self.updateSegmentInfo(null, true);
                }
            });
        });
        document.getElementById('kptThreshold')?.addEventListener('change', () => {
            if (!VIEWER_STATE.imageMode) {
                self.updateSegmentInfo(null, true);
            }
        });
    }

    async loadVideo(index) {
        if (index < 0 || index >= this.detections.length) return;

        this.currentVideoIndex = index;
        this.currentSegmentIndex = 0;
        this.isSeeking = false;
        this.pendingDelta = 0;
        this.loopGen++;
        this.lastDrawnFrame = -1;  // Reset frame tracker for new video

        const det = this.detections[index];
        const videoPlayer = document.getElementById('videoPlayer');
        const canvas = document.getElementById('overlayCanvas');
        const ctx = canvas?.getContext('2d');

        if (!videoPlayer || !canvas || !ctx) return;

        // Reset container styles so it adapts to the new video
        const container = document.getElementById('videoContainer');
        if (container) {
            container.style.width = '';
            container.style.height = '';
            container.style.aspectRatio = '';
        }

        // Set default FPS (will be updated after JSON loads)
        this.fps = det.video_info?.fps || 30;
        VIEWER_STATE.fps = this.fps;

        // Update title
        const sourcePath = det.source_video || '';
        const fileName = det.isFaceCrop ? (det._fileName || 'unknown') : (sourcePath.split(/[/\\]/).pop() || det._fileName || `Video ${index + 1}`);
        
        const titleEl = document.getElementById('videoTitle');
        if (titleEl) titleEl.textContent = `Video ${index + 1}/${this.detections.length}: ${fileName}`;
        
        const metaEl = document.getElementById('videoMeta');
        const info = det.video_info || { width: 640, height: 480, duration_seconds: 0 };
        if (metaEl) {
            const width = info.width || 640;
            const height = info.height || 480;
            const duration = info.duration_seconds || 0;
            metaEl.textContent = `${width}x${height} | ${this.fps.toFixed(1)} fps | ${duration.toFixed(1)}s`;
        }

        // Find video file - use video_path from item if available
        let videoUrl = null;
        
        // Primary: Use video_path from data_index.json
        if (det.video_path) {
            videoUrl = det.video_path;
            console.log('[VIDEO] Using video_path from index:', videoUrl);
        }
        // Fallback: Try remote URL
        else if (det._remoteVideoUrl) {
            videoUrl = det._remoteVideoUrl;
            console.log('[VIDEO] Using remote URL:', videoUrl);
        }
        // Fallback: Search in videoFiles dictionary
        else if (this.videoFiles[fileName]) {
            videoUrl = this.videoFiles[fileName];
            console.log('[VIDEO] Found in videoFiles:', videoUrl);
        }
        // Fallback: Construct path from source_video
        else if (det.source_video) {
            // Extract filename and construct path
            const sourceFile = det.source_video.split(/[/\\]/).pop();
            videoUrl = `data_link/extracted_person_clips/${sourceFile}`;
            console.log('[VIDEO] Constructed from source_video:', videoUrl);
        }
        // Last fallback: Try to construct from _fileName
        else if (det._fileName) {
            const baseName = det._fileName.replace(/\.json$/i, '');
            videoUrl = `data_link/extracted_person_clips/${baseName}.mp4`;
            console.log('[VIDEO] Constructed from _fileName:', videoUrl);
        }

        if (videoUrl) {
            console.log('[VIDEO] Setting src to:', videoUrl);
            videoPlayer.src = videoUrl;
            videoPlayer.load();
            videoPlayer.style.opacity = '1';
            
            const playBtn = document.getElementById('playPause');
            if (playBtn) {
                playBtn.disabled = false;
                playBtn.textContent = '▶ Play';
            }
            
            if (det.isClip) videoPlayer.loop = true;
            else videoPlayer.loop = false;

            videoPlayer.onerror = () => {
                console.warn("Video load error for", fileName, "- tried URL:", videoUrl);
            };
        } else {
            console.warn('[VIDEO] No video URL found for:', det);
            videoPlayer.removeAttribute('src');
            videoPlayer.load();
            videoPlayer.style.opacity = '0.3';
            
            const playBtn = document.getElementById('playPause');
            if (playBtn) {
                playBtn.disabled = true;
                playBtn.textContent = 'No Video Loaded';
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#fff";
            ctx.font = "20px Segoe UI";
            ctx.fillText("No video file loaded.", 50, 50);
        }

        // Load detection JSON file
        if (det.json_path) {
            try {
                console.log('[DETECTIONS] Loading JSON from:', det.json_path);
                const response = await fetch(det.json_path, { cache: 'no-store' });
                if (response.ok) {
                    const jsonData = await response.json();
                    console.log('[DETECTIONS] JSON loaded, frame_data entries:', Object.keys(jsonData.frame_data || {}).length);
                    
                    // Copy video_info from JSON (items in data_index don't include it)
                    if (jsonData.video_info && !det.video_info) {
                        det.video_info = jsonData.video_info;
                    }
                    
                    // Store parent_clip reference for transcription lookup (face crops)
                    if (jsonData.parent_clip) {
                        det.parent_clip = jsonData.parent_clip;
                    }

                    // Calculate FPS from frame data if not provided
                    let fps = jsonData.fps || det.video_info?.fps;
                    if (!fps && jsonData.start_frame !== undefined && jsonData.end_frame !== undefined &&
                        jsonData.start_seconds !== undefined && jsonData.end_seconds !== undefined) {
                        const frameCount = jsonData.end_frame - jsonData.start_frame;
                        const duration = jsonData.end_seconds - jsonData.start_seconds;
                        if (duration > 0) {
                            fps = frameCount / duration;
                            console.log(`[DETECTIONS] Calculated FPS: ${fps.toFixed(2)}`);
                        }
                    }

                    // Store FPS in det.video_info for later use
                    if (fps) {
                        if (!det.video_info) det.video_info = {};
                        det.video_info.fps = fps;
                        console.log(`[DETECTIONS] Stored FPS: ${fps.toFixed(2)}`);
                    }

                    // For clips, duration is the clip duration, not the source video duration
                    if (det.isClip && det.video_info && jsonData.start_seconds !== undefined && jsonData.end_seconds !== undefined) {
                        det.video_info = { ...det.video_info, duration_seconds: jsonData.end_seconds - jsonData.start_seconds };
                    }
                    
                    // Store the detections in segments
                    // For person clips: one segment = the clip itself
                    // For full videos: multiple segments as defined
                    if (det.isClip || !det.segments) {
                        // Create a single segment with the entire clip's detection data
                        det.segments = [{
                            start_seconds: jsonData.start_seconds || 0,
                            end_seconds: jsonData.end_seconds || 0,
                            start_frame: jsonData.start_frame || 0,
                            end_frame: jsonData.end_frame || 0,
                            max_persons: jsonData.max_persons || 0,
                            frame_data: jsonData.frame_data || {}
                        }];
                    } else {
                        // For multi-segment videos: add frame_data to each segment
                        det.segments.forEach(seg => {
                            seg.frame_data = jsonData.frame_data || {};
                        });
                    }
                    console.log('[DETECTIONS] Segments updated with frame_data');
                } else {
                    console.warn('[DETECTIONS] Failed to load JSON:', response.status);
                }
            } catch (err) {
                console.error('[DETECTIONS] Error loading detection JSON:', err);
            }
        }

        // Update FPS if it was calculated from JSON
        if (det.video_info?.fps) {
            this.fps = det.video_info.fps;
            VIEWER_STATE.fps = this.fps;
            console.log(`[VIDEO] Updated FPS to: ${this.fps.toFixed(2)}`);
            
            // Update UI with new FPS
            const metaEl = document.getElementById('videoMeta');
            if (metaEl && det.video_info) {
                const width = det.video_info.width || 640;
                const height = det.video_info.height || 480;
                const duration = det.video_info.duration_seconds || 0;
                metaEl.textContent = `${width}x${height} | ${this.fps.toFixed(1)} fps | ${duration.toFixed(1)}s`;
            }
        }

        // Load quality metrics (MagFace and OFIQ)
        const qualityEl = document.getElementById('qualityMetrics');
        if (qualityEl) {
            qualityEl.innerHTML = '';
            
            // Load MagFace data
            if (det.magface_path) {
                try {
                    const response = await fetch(det.magface_path, { cache: 'no-store' });
                    if (response.ok) {
                        const magfaceData = await response.json();
                        if (magfaceData.unified_score) {
                            const score = magfaceData.unified_score;
                            const line = document.createElement('div');
                            line.innerHTML = `<strong>MagFace:</strong> max=${score.max?.toFixed(2) || '-'} | mean=${score.mean?.toFixed(2) || '-'} | p50=${score.p50?.toFixed(2) || '-'}`;
                            qualityEl.appendChild(line);
                        }
                    }
                } catch (err) {
                    console.warn('[QUALITY] Error loading MagFace data:', err);
                }
            }
            
            // Load OFIQ data
            if (det.ofiq_attr_path) {
                try {
                    const response = await fetch(det.ofiq_attr_path, { cache: 'no-store' });
                    if (response.ok) {
                        const ofiqData = await response.json();
                        const measures = ['sharpness', 'compression_artifacts', 'expression_neutrality', 'no_head_coverings', 'face_occlusion_prevention'];
                        
                        for (const measure of measures) {
                            if (ofiqData[measure]) {
                                const data = ofiqData[measure];
                                const line = document.createElement('div');
                                line.innerHTML = `<strong>${measure}:</strong> max=${data.max?.toFixed(2) || '-'} | mean=${data.mean?.toFixed(2) || '-'}`;
                                qualityEl.appendChild(line);
                            }
                        }
                    }
                } catch (err) {
                    console.warn('[QUALITY] Error loading OFIQ data:', err);
                }
            }
        }

        // Load transcription data
        const transcriptionEl = document.getElementById('transcriptionText');
        this.transcriptionSegments = [];  // Reset segments
        if (transcriptionEl) {
            transcriptionEl.innerHTML = '';
            transcriptionEl.style.display = 'none';
            
            // Try direct transcription_path first (person clips)
            let transcriptionPath = det.transcription_path;
            
            // For face crops, try to load from parent clip
            if (!transcriptionPath && det.parent_clip?.file) {
                // Construct path: parent clip is in extracted_person_clips folder
                // parent_clip.file is like "ClipName.mp4" or "ClipName.json"
                const parentFile = det.parent_clip.file.replace(/\.(mp4|json)$/i, '');
                transcriptionPath = `data_link/extracted_person_clips/${parentFile}.transcription.json`;
                console.log('[TRANSCRIPTION] Trying parent clip transcription:', transcriptionPath);
            }
            
            if (transcriptionPath) {
                try {
                    const response = await fetch(transcriptionPath, { cache: 'no-store' });
                    if (response.ok) {
                        const transData = await response.json();
                        // Store segments for subtitle sync
                        if (transData.segments && Array.isArray(transData.segments)) {
                            this.transcriptionSegments = transData.segments;
                            console.log('[TRANSCRIPTION] Loaded', this.transcriptionSegments.length, 'segments');
                        }
                        if (transData.transcription) {
                            const langLabel = transData.language ? ` [${transData.language}]` : '';
                            transcriptionEl.innerHTML = `<strong>Transcription${langLabel}:</strong><br>${transData.transcription}`;
                            transcriptionEl.style.display = 'block';
                            console.log('[TRANSCRIPTION] Loaded transcription, length:', transData.transcription.length);
                        }
                    }
                } catch (err) {
                    console.warn('[TRANSCRIPTION] Error loading transcription data:', err);
                }
            }
        }
        
        // Create/update subtitle overlay
        this._ensureSubtitleOverlay();

        // Update segments
        const segmentNav = document.getElementById('segmentNav');
        if (segmentNav) {
            segmentNav.innerHTML = '';
            (det.segments || []).forEach((seg, i) => {
                const btn = document.createElement('button');
                btn.className = 'segment-btn' + (i === 0 ? ' active' : '');
                btn.textContent = `${i + 1}: ${formatTime(seg.start_seconds)} - ${formatTime(seg.end_seconds)}`;
                btn.onclick = () => this.jumpToSegment(i);
                segmentNav.appendChild(btn);
            });
        }

        // Update list highlighting
        document.querySelectorAll('#videoList .video-item').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });

        this.jumpToSegment(0);
    }

    jumpToSegment(index) {
        const det = this.detections[this.currentVideoIndex];
        const segments = det?.segments || [];
        if (index < 0 || index >= segments.length) return;

        this.currentSegmentIndex = index;
        this.lastDrawnFrame = -1;  // Reset frame tracker when jumping to new segment
        const seg = segments[index];
        const videoPlayer = document.getElementById('videoPlayer');

        if (videoPlayer) {
            // For clips, start_seconds is the position in the original video — clip file starts at 0
            videoPlayer.currentTime = det?.isClip ? 0 : seg.start_seconds;
        }

        // Update segment buttons
        document.querySelectorAll('.segment-btn').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });

        // Draw detections immediately
        this.updateSegmentInfo(null, true);
    }

    drawDetections(segment, currentFrame) {
        const canvas = document.getElementById('overlayCanvas');
        const videoPlayer = document.getElementById('videoPlayer');
        const ctx = canvas?.getContext('2d');

        if (!ctx || !canvas || !videoPlayer) return;

        // Skip if we already drew this frame
        if (currentFrame === this.lastDrawnFrame) {
            return;
        }
        this.lastDrawnFrame = currentFrame;

        // Clear to transparent — native <video> element (opacity:1, z-index:0) shows through.
        // Do NOT drawImage(videoPlayer): on Linux hardware-decoded video, ctx.drawImage returns
        // stale/black frames during active playback, hiding the video while audio plays.
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!segment?.frame_data) {
            return;
        }

        const det = this.detections[this.currentVideoIndex];
        if (!det) return;

        // Calculate scale factors
        const videoWidth = canvas.videoWidth || det.video_info?.width || 1920;
        const videoHeight = canvas.videoHeight || det.video_info?.height || 1080;
        const scaleX = canvas.width / videoWidth;
        const scaleY = canvas.height / videoHeight;

        let lookupFrame = currentFrame;
        if (det.isClip) {
            lookupFrame = (segment.start_frame || 0) + currentFrame;
        }

        // Try exact frame lookup first
        let framePeople = segment.frame_data[lookupFrame] || segment.frame_data[lookupFrame.toString()];

        // If exact frame not found, find nearest neighbor
        if (!framePeople || !Array.isArray(framePeople)) {
            const availableFrames = Object.keys(segment.frame_data)
                .map(k => parseInt(k))
                .filter(f => !isNaN(f));

            if (availableFrames.length > 0) {
                // Find closest frame
                let closestFrame = availableFrames[0];
                let minDiff = Math.abs(availableFrames[0] - lookupFrame);

                for (let f of availableFrames) {
                    const diff = Math.abs(f - lookupFrame);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestFrame = f;
                    }
                }

                // Only use if within reasonable range (1 frame tolerance)
                if (minDiff <= 1) {
                    framePeople = segment.frame_data[closestFrame];
                }
            }
        }
        if (!framePeople || !Array.isArray(framePeople)) {
            // No detections for this frame
            return;
        }

        const showBBoxCheck = document.getElementById('showBBox');
        const showKeypointsCheck = document.getElementById('showKeypoints');
        const showScoresCheck = document.getElementById('showScores');
        const showIdsCheck = document.getElementById('showIds');
        const showFaceCropArcfaceCheck = document.getElementById('showFaceCropArcface');
        const showFaceCropOfiqCheck = document.getElementById('showFaceCropOfiq');
        const kptThresholdInput = document.getElementById('kptThreshold');

        const showBBox = showBBoxCheck?.checked;
        const showKeypoints = showKeypointsCheck?.checked;
        const showScores = showScoresCheck?.checked;
        const showIds = showIdsCheck?.checked;
        const KPT_THRESHOLD = parseFloat(kptThresholdInput?.value) || 0.1;

        framePeople.forEach(person => {
            const color = getColorForId(person.track_id || 0);

            // Draw Bounding Box (scaled)
            if (showBBox && person.bbox) {
                const [x1, y1, x2, y2] = person.bbox;
                const sx1 = x1 * scaleX;
                const sy1 = y1 * scaleY;
                const w = (x2 - x1) * scaleX;
                const h = (y2 - y1) * scaleY;

                ctx.strokeStyle = color;
                ctx.lineWidth = VIEWER_CONFIG.LINE_WIDTH_BBOX;
                ctx.strokeRect(sx1, sy1, w, h);

                if (showIds || showScores) {
                    let label = "";
                    if (showIds) label += `ID: ${person.track_id} `;
                    if (showScores) label += `${(person.score * 100).toFixed(0)}%`;

                    ctx.font = VIEWER_CONFIG.FONT;
                    const textMetrics = ctx.measureText(label);
                    const bgHeight = 20;
                    const bgWidth = textMetrics.width + 10;

                    ctx.fillStyle = color;
                    ctx.fillRect(sx1, sy1 - bgHeight, bgWidth, bgHeight);
                    ctx.fillStyle = "#000";
                    ctx.fillText(label, sx1 + 5, sy1 - 5);
                }
            }

            // Draw face crop corners (scaled)
            if (showFaceCropArcfaceCheck?.checked && person.face_crop_corners_arcface) {
                this.drawCropQuad(ctx, person.face_crop_corners_arcface, '#ffcc00', scaleX, scaleY);
            }
            if (showFaceCropOfiqCheck?.checked && person.face_crop_corners_ofiq) {
                this.drawCropQuad(ctx, person.face_crop_corners_ofiq, '#cc66ff', scaleX, scaleY);
            }

            // Draw Keypoints (scaled)
            if (showKeypoints && person.keypoints) {
                const kpts = person.keypoints;
                const scores = person.keypoint_scores || [];

                // Exclude keypoints clamped to frame boundary (pose model artifact when body is off-screen)
                const EDGE_MARGIN = 2;
                const isInBounds = (p) => {
                    const sx = p[0] * scaleX;
                    const sy = p[1] * scaleY;
                    return sx > EDGE_MARGIN && sx < canvas.width - EDGE_MARGIN && sy > EDGE_MARGIN && sy < canvas.height - EDGE_MARGIN;
                };

                // Draw skeleton
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = VIEWER_CONFIG.LINE_WIDTH_SKELETON;
                VIEWER_CONFIG.SKELETON.forEach(([i, j]) => {
                    if (i < kpts.length && j < kpts.length) {
                        const p1 = kpts[i];
                        const p2 = kpts[j];
                        const s1 = scores[i] ?? 1;
                        const s2 = scores[j] ?? 1;
                        if (isInBounds(p1) && isInBounds(p2) && s1 >= KPT_THRESHOLD && s2 >= KPT_THRESHOLD) {
                            ctx.moveTo(p1[0] * scaleX, p1[1] * scaleY);
                            ctx.lineTo(p2[0] * scaleX, p2[1] * scaleY);
                        }
                    }
                });
                ctx.stroke();

                // Draw points
                kpts.forEach((p, idx) => {
                    const s = scores[idx] ?? 1;
                    if (isInBounds(p) && s >= KPT_THRESHOLD) {
                        ctx.beginPath();
                        ctx.fillStyle = "#fff";
                        ctx.arc(p[0] * scaleX, p[1] * scaleY, VIEWER_CONFIG.KEYPOINT_RADIUS, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                });
            }
        });
    }

    drawCropQuad(ctx, corners, color, scaleX = 1, scaleY = 1) {
        if (!corners || corners.length < 4) return;
        ctx.beginPath();
        ctx.moveTo(corners[0][0] * scaleX, corners[0][1] * scaleY);
        for (let ci = 1; ci < corners.length; ci++) {
            ctx.lineTo(corners[ci][0] * scaleX, corners[ci][1] * scaleY);
        }
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = color;
        corners.forEach(([cx, cy]) => {
            ctx.beginPath();
            ctx.arc(cx * scaleX, cy * scaleY, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    updateSegmentInfo(metadata, drawCanvas = true) {
        if (VIEWER_STATE.imageMode) return;

        const det = this.detections[this.currentVideoIndex];
        const segments = det?.segments || [];
        const seg = segments[this.currentSegmentIndex];
        
        if (!det || !seg) return;

        const videoPlayer = document.getElementById('videoPlayer');
        const time = videoPlayer?.currentTime || 0;
        // Use standard rounding instead of floor for more accurate frame number
        const currentFrame = Math.round(time * this.fps);

        let displayFrame = currentFrame;
        if (det.isClip) {
            displayFrame = seg.start_frame + currentFrame;
        }

        const frameInfoEl = document.getElementById('frameInfo');
        const segmentInfoEl = document.getElementById('segmentInfo');
        
        if (frameInfoEl) {
            if (det.isClip) {
                frameInfoEl.textContent = `Clip Frame: ${currentFrame} (Abs: ${displayFrame})`;
            } else {
                frameInfoEl.textContent = `Frame: ${currentFrame} (${time.toFixed(2)}s)`;
            }
        }

        if (segmentInfoEl) {
            if (det.isClip) {
                segmentInfoEl.textContent = `Clip: ${det._fileName}`;
            } else {
                const segStartFrame = Math.floor(seg.start_seconds * this.fps + 0.5);
                const segEndFrame = Math.floor(seg.end_seconds * this.fps + 0.5);
                segmentInfoEl.textContent = `Segment ${this.currentSegmentIndex + 1}/${segments.length} | ${seg.max_persons} person(s) | Frames ${segStartFrame}-${segEndFrame}`;
            }
        }

        // Update progress
        const segmentProgress = document.getElementById('segmentProgress');
        if (segmentProgress) {
            let progress = 0;
            if (det.isClip) {
                progress = videoPlayer && videoPlayer.duration ? (time / videoPlayer.duration) : 0;
            } else {
                const segDuration = seg.end_seconds - seg.start_seconds;
                progress = Math.max(0, Math.min(1, (time - seg.start_seconds) / segDuration));
            }
            segmentProgress.value = Math.round(progress * 1000);
        }

        if (drawCanvas) this.drawDetections(seg, currentFrame);
        
        // Update subtitles overlay
        this.updateSubtitle(time);
    }

    doSeek(delta) {
        this.isSeeking = true;
        const videoPlayer = document.getElementById('videoPlayer');
        const frameDuration = 1 / this.fps;
        if (videoPlayer) {
            videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime + delta * frameDuration);
            // Force immediate redraw after seek
            this.lastDrawnFrame = -1;  // Reset frame tracking to force redraw
        }
    }

    stepFrame(delta) {
        const videoPlayer = document.getElementById('videoPlayer');
        if (videoPlayer) videoPlayer.pause();
        
        if (!this.isSeeking) {
            this.doSeek(delta);
            // Immediately update canvas - don't wait for RAF
            setTimeout(() => {
                this.updateSegmentInfo(null, true);
            }, 16);  // ~1 frame at 60fps to let video buffer update
        } else {
            this.pendingDelta += delta;
        }
    }

    nextVideo() {
        this.loadVideo((this.currentVideoIndex + 1) % this.detections.length);
    }

    prevVideo() {
        this.loadVideo((this.currentVideoIndex - 1 + this.detections.length) % this.detections.length);
    }


    onProgressChange() {
        const det = this.detections[this.currentVideoIndex];
        const seg = det?.segments?.[this.currentSegmentIndex];
        const segmentProgress = document.getElementById('segmentProgress');
        const videoPlayer = document.getElementById('videoPlayer');
        
        if (!seg || !segmentProgress || !videoPlayer) return;
        
        const fraction = segmentProgress.value / 1000;
        if (det.isClip) {
            videoPlayer.currentTime = fraction * (videoPlayer.duration || 0);
        } else {
            const segDuration = seg.end_seconds - seg.start_seconds;
            videoPlayer.currentTime = seg.start_seconds + fraction * segDuration;
        }
    }

    startRenderLoop() {
        const self = this;
        const loopGen = this.loopGen;
        const videoPlayer = document.getElementById('videoPlayer');
        
        // Cancel any existing animation frame
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        // Remove old timeupdate listener if it exists
        if (videoPlayer && this.timeupdateHandler) {
            videoPlayer.removeEventListener('timeupdate', this.timeupdateHandler);
            this.timeupdateHandler = null;
        }
        
        let lastUpdateTime = -Infinity;  // Track last video time we processed
        
        const renderFrame = () => {
            // Check if video is still playing - use videoPlayer.paused as the primary check
            // loopGen is just a sanity check to ensure we're still on the same video
            if (videoPlayer?.paused || loopGen !== self.loopGen) {
                return;
            }
            
            // Get current video time
            const currentTime = videoPlayer?.currentTime || 0;
            // Detect loop restart (time jumped backward) and reset tracking
            if (currentTime < lastUpdateTime) {
                lastUpdateTime = -Infinity;
                self.lastDrawnFrame = -1;
            }
            // Only update if video time has advanced enough (at least 1 frame duration)
            const frameDuration = 1 / self.fps;
            if (currentTime - lastUpdateTime >= frameDuration * 0.8) {  // 0.8 factor for tolerance
                self.updateSegmentInfo(null, true);
                lastUpdateTime = currentTime;
            }
            
            // Always request next frame to keep monitor smooth (60fps)
            self.animationFrameId = requestAnimationFrame(renderFrame);
        };
        
        // Start render loop
        // RAF keeps monitor smooth at 60fps, but only redraws when video frame actually changes
        self.animationFrameId = requestAnimationFrame(renderFrame);
    }

    /**
     * Create or get the subtitle overlay element
     */
    _ensureSubtitleOverlay() {
        let overlay = document.getElementById('subtitleOverlay');
        if (!overlay) {
            const container = document.getElementById('videoContainer');
            if (container) {
                overlay = document.createElement('div');
                overlay.id = 'subtitleOverlay';
                overlay.style.cssText = `
                    position: absolute;
                    bottom: 50px;
                    left: 50%;
                    transform: translateX(-50%);
                    max-width: 90%;
                    padding: 8px 16px;
                    background: rgba(0, 0, 0, 0.75);
                    color: #fff;
                    font-size: 1.2rem;
                    font-weight: 500;
                    text-align: center;
                    border-radius: 4px;
                    z-index: 100;
                    pointer-events: none;
                    text-shadow: 1px 1px 2px #000;
                    display: none;
                `;
                container.appendChild(overlay);
            }
        }
        return overlay;
    }

    /**
     * Update the subtitle overlay based on current video time
     */
    updateSubtitle(currentTime) {
        const overlay = document.getElementById('subtitleOverlay');
        if (!overlay) return;
        
        const showSubtitles = document.getElementById('showSubtitles')?.checked;
        if (!showSubtitles || !this.transcriptionSegments.length) {
            overlay.style.display = 'none';
            return;
        }
        
        // Find the segment that matches current time
        const segment = this.transcriptionSegments.find(seg => 
            currentTime >= seg.start && currentTime < seg.end
        );
        
        if (segment && segment.text) {
            overlay.textContent = segment.text;
            overlay.style.display = 'block';
        } else {
            overlay.style.display = 'none';
        }
    }
}

// Helper function
function formatTime(seconds) {
    if (seconds == null) return '?';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// Export for use
window.VideoViewer = VideoViewer;
