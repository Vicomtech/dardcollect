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
        if (videoList) {
            videoList.addEventListener('click', (e) => {
                const item = e.target.closest('.video-item');
                if (item) {
                    const idx = parseInt(item.dataset.index);
                    if (idx >= 0 && idx < this.items.length) {
                        this.loadImage(idx);
                    }
                }
            });
        }

        // Button handlers
        document.getElementById('prevImage')?.addEventListener('click', () => {
            if (this.currentIdx > 0) this.loadImage(this.currentIdx - 1);
        });
        
        document.getElementById('nextImage')?.addEventListener('click', () => {
            if (this.currentIdx < this.items.length - 1) this.loadImage(this.currentIdx + 1);
        });

        // Keyboard handlers
        document.addEventListener('keydown', (e) => {
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
        });

        // Viz control handlers
        const vizControls = [
            document.getElementById('showBBox'),
            document.getElementById('showKeypoints'),
            document.getElementById('showScores'),
            document.getElementById('kptThreshold')
        ];
        vizControls.forEach(ctrl => {
            ctrl?.addEventListener('change', () => {
                if (VIEWER_STATE.imageMode && this.currentData) {
                    this.loadImage(this.currentIdx);
                }
            });
        });
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

            img.onload = () => {
                if (myGen !== this._loadGen || !VIEWER_STATE.imageMode) return;
                this.drawImage(img, data, canvas, ctx);
            };
            img.onerror = () => this.handleImageError(canvas, ctx, data.image_path);

            img.src = `data_link/archive_org_public_domain/images/${data.image_path}`;
        } catch (err) {
            console.error('Error loading image:', err);
            ctx.fillStyle = '#f00';
            ctx.font = '16px Segoe UI';
            ctx.fillText('Error: ' + err.message, 20, 40);
        }
    }

    drawImage(img, data, canvas, ctx) {
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
        
        // Draw detections
        const scaleX = displayWidth / img.naturalWidth;
        const scaleY = displayHeight / img.naturalHeight;
        
        if (data.detections && data.detections.length > 0) {
            if (typeof window.drawImageDetectionsScaled === 'function') {
                window.drawImageDetectionsScaled(ctx, data.detections, scaleX, scaleY);
            }
        }
        
        // Update UI
        document.getElementById('videoTitle').textContent = data.image_path || 'Image';
        document.getElementById('videoMeta').textContent = 
            `${data.image_size?.width || '?'}×${data.image_size?.height || '?'} | ${data.detections?.length || 0} persons`;
        
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
