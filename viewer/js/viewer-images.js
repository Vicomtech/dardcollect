/**
 * Image Viewer Module
 * Handles loading, displaying, and navigating image detections
 */

class ImageViewer {
    constructor() {
        this.items = [];
        this.currentIdx = 0;
        this.currentData = null;
        this.mode = 'image';
        this._loadGen = 0;
        this._handlers = {};
    }

    resolveSourceImageUrl(imagePath) {
        const normalized = String(imagePath || '').replace(/\\/g, '/');
        if (!normalized) return '';
        if (normalized.startsWith('data_link/')) return normalized;
        if (normalized.startsWith('DARD/')) {
            return `data_link/${normalized.slice('DARD/'.length)}`;
        }
        if (normalized.startsWith('/')) {
            return `data_link${normalized}`;
        }
        return `data_link/${normalized}`;
    }

    async init(items) {
        ViewerCommon.initializeDomElements();
        
        this.items = items;
        this.currentIdx = 0;
        VIEWER_STATE.imageMode = true;
        VIEWER_STATE.imageDetections = items;
        
        ViewerCommon.showViewer();
        ViewerCommon.toggleUiVisibility('image');
        
        this.buildImageList();
        this.attachEventHandlers();
        
        if (items.length > 0) {
            await this.loadImage(0);
        }
    }

    destroy() {
        // Remove DOM event listeners
        if (this._handlers.documentKeydown) {
            document.removeEventListener('keydown', this._handlers.documentKeydown);
        }
        if (this._handlers.prevClick) {
            document.getElementById('prevImage')?.removeEventListener('click', this._handlers.prevClick);
        }
        if (this._handlers.nextClick) {
            document.getElementById('nextImage')?.removeEventListener('click', this._handlers.nextClick);
        }
        if (this._handlers.vizChange) {
            const checkboxes = [
                document.getElementById('showBBox'),
                document.getElementById('showKeypoints'),
                document.getElementById('showScores'),
                document.getElementById('showIds'),
                document.getElementById('showFaceCropArcface'),
                document.getElementById('showFaceCropOfiq')
            ];
            checkboxes.forEach(ctrl => {
                ctrl?.removeEventListener('click', this._handlers.vizChange);
            });
            document.getElementById('kptThreshold')?.removeEventListener('change', this._handlers.vizChange);
        }
        if (this._handlers.listClick) {
            document.getElementById('videoList')?.removeEventListener('click', this._handlers.listClick);
        }
    }

    buildImageList() {
        const videoList = document.getElementById('videoList');
        if (!videoList) return;
        
        videoList.innerHTML = '';
        this.items.forEach((item, i) => {
            const div = document.createElement('div');
            div.className = 'video-item' + (i === 0 ? ' active' : '');
            div.textContent = item.json_path?.split('/').pop()?.replace('.json', '') || `Image ${i}`;
            div.dataset.index = i;
            videoList.appendChild(div);
        });
        
        document.getElementById('videoCount').textContent = this.items.length;
    }

    attachEventHandlers() {
        const videoList = document.getElementById('videoList');
        this._handlers.listClick = (e) => {
            const el = e.target.closest('.video-item');
            if (el) {
                const idx = parseInt(el.dataset.index);
                if (idx >= 0 && idx < this.items.length) {
                    this.loadImage(idx);
                }
            }
        };
        if (videoList) {
            videoList.addEventListener('click', this._handlers.listClick);
        }

        // Button handlers
        this._handlers.prevClick = () => {
            if (this.currentIdx > 0) this.loadImage(this.currentIdx - 1);
        };
        this._handlers.nextClick = () => {
            if (this.currentIdx < this.items.length - 1) this.loadImage(this.currentIdx + 1);
        };
        document.getElementById('prevImage')?.addEventListener('click', this._handlers.prevClick);
        document.getElementById('nextImage')?.addEventListener('click', this._handlers.nextClick);

        // Keyboard handlers
        this._handlers.documentKeydown = (e) => {
            if (VIEWER_STATE.imageMode) {
                if (e.key === 'ArrowLeft') { 
                    e.preventDefault(); 
                    if (this.currentIdx > 0) this.loadImage(this.currentIdx - 1);
                }
                if (e.key === 'ArrowRight') { 
                    e.preventDefault(); 
                    if (this.currentIdx < this.items.length - 1) this.loadImage(this.currentIdx + 1);
                }
            }
        };
        document.addEventListener('keydown', this._handlers.documentKeydown);

        // Viz control handlers - use 'click' for checkboxes for reliability
        this._handlers.vizChange = () => {
            if (VIEWER_STATE.imageMode && this.currentData) {
                this.loadImage(this.currentIdx);
            }
        };
        const checkboxes = [
            document.getElementById('showBBox'),
            document.getElementById('showKeypoints'),
            document.getElementById('showScores'),
            document.getElementById('showIds'),
            document.getElementById('showFaceCropArcface'),
            document.getElementById('showFaceCropOfiq')
        ];
        checkboxes.forEach(ctrl => {
            ctrl?.addEventListener('click', this._handlers.vizChange);
        });
        document.getElementById('kptThreshold')?.addEventListener('change', this._handlers.vizChange);
    }

    async loadImage(idx) {
        if (idx < 0 || idx >= this.items.length) return;

        this._loadGen++;
        const myGen = this._loadGen;

        this.currentIdx = idx;
        VIEWER_STATE.currentImageIdx = idx;

        const item = this.items[idx];
        const canvas = document.getElementById('imageCanvas');
        const ctx = canvas?.getContext('2d');

        if (!canvas || !ctx) return;

        // Prevent stale draws from previous viewer instances
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        try {
            const response = await fetch(item.json_path, { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            // Abort if a newer load started or viewer switched away from image mode
            if (myGen !== this._loadGen || !VIEWER_STATE.imageMode) return;

            this.currentData = data;
            VIEWER_STATE.currentImageData = data;

            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = async () => {
                if (myGen !== this._loadGen || !VIEWER_STATE.imageMode) return;
                if (typeof currentViewer !== 'undefined' && currentViewer !== this) return;
                await this.drawImage(img, data, canvas, ctx, item.type, item);
            };

            if (item.type === 'image_face_crop') {
                img.onerror = () => this.handleImageError(canvas, ctx, item.image_path);
                img.src = item.image_path;
            } else {
                img.onerror = () => this.handleImageError(canvas, ctx, data.image_path);
                img.src = this.resolveSourceImageUrl(data.image_path);
            }
        } catch (err) {
            console.error('Error loading image:', err);
            ctx.fillStyle = '#f00';
            ctx.font = '16px Segoe UI';
            ctx.fillText('Error: ' + err.message, 20, 40);
        }
    }

    async drawImage(img, data, canvas, ctx, itemType, item) {
        // Calculate scaling
        const maxHeight = window.innerHeight * 0.65;
        const maxWidth = window.innerWidth - 100;
        const aspectRatio = img.naturalWidth / img.naturalHeight;

        let displayWidth = img.naturalWidth;
        let displayHeight = img.naturalHeight;

        if (displayHeight > maxHeight) {
            displayHeight = maxHeight;
            displayWidth = displayHeight * aspectRatio;
        }
        if (displayWidth > maxWidth) {
            displayWidth = maxWidth;
            displayHeight = displayWidth / aspectRatio;
        }

        // Set canvas
        const scale = window.devicePixelRatio || 1;
        canvas.width = displayWidth * scale;
        canvas.height = displayHeight * scale;
        canvas.style.width = displayWidth + 'px';
        canvas.style.height = displayHeight + 'px';
        ctx.scale(scale, scale);

        // Draw image
        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

        // Draw detections / keypoints
        // Use data.image_size for scaling: detections are in original image coordinate space,
        // which may differ from the actual loaded image file dimensions.
        const srcWidth = data.image_size?.width || img.naturalWidth;
        const srcHeight = data.image_size?.height || img.naturalHeight;
        const scaleX = displayWidth / srcWidth;
        const scaleY = displayHeight / srcHeight;

        if (itemType === 'image_face_crop') {
            // Face crop: keypoints are already in 616×616 OFIQ space, draw directly
            if (data.keypoints && data.keypoint_scores) {
                const syntheticDetection = {
                    person_id: 0,
                    keypoints: data.keypoints,
                    keypoint_scores: data.keypoint_scores,
                };
                // Face crop keypoints are in 616x616 space regardless of actual image size
                const cropScaleX = displayWidth / 616;
                const cropScaleY = displayHeight / 616;
                if (typeof window.drawImageDetectionsScaled === 'function') {
                    window.drawImageDetectionsScaled(ctx, [syntheticDetection], cropScaleX, cropScaleY);
                }
            }
            const quality = data.quality_score != null ? ` | quality: ${data.quality_score.toFixed(2)}` : '';
            document.getElementById('videoTitle').textContent = data.uuid || 'Face Crop';
            document.getElementById('videoMeta').textContent = `616×616 OFIQ face crop${quality}`;
        } else {
            if (data.detections && data.detections.length > 0) {
                if (typeof window.drawImageDetectionsScaled === 'function') {
                    window.drawImageDetectionsScaled(ctx, data.detections, scaleX, scaleY);
                }
            }
            const imageName = String(data.image_path || '').split(/[\\/]/).pop() || 'Image';
            document.getElementById('videoTitle').textContent = imageName;
            document.getElementById('videoMeta').textContent =
                `${data.image_size?.width || '?'}×${data.image_size?.height || '?'} | ${data.detections?.length || 0} persons`;
        }

        // Load quality metrics (MagFace and OFIQ)
        const qualityEl = document.getElementById('qualityMetrics');
        if (qualityEl) {
            qualityEl.innerHTML = '';

            // Load MagFace data
            if (item.magface_path) {
                try {
                    const response = await fetch(item.magface_path, { cache: 'no-store' });
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
            if (item.ofiq_attr_path) {
                try {
                    const response = await fetch(item.ofiq_attr_path, { cache: 'no-store' });
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
        // Update list
        document.querySelectorAll('#videoList .video-item').forEach((el, i) => {
            el.classList.toggle('active', i === this.currentIdx);
        });
    }

    handleImageError(canvas, ctx, path) {
        console.error('Failed to load image:', path);
        ctx.fillStyle = '#f00';
        ctx.font = '16px Segoe UI';
        ctx.fillText('Failed to load: ' + path, 20, 40);
    }
}

// Export for use
window.ImageViewer = ImageViewer;
