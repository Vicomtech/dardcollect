/**
 * Document Viewer Module
 * Handles loading and displaying preprocessed documents (annotation + text)
 */

class DocumentViewer {
    constructor() {
        this.items = [];
        this.currentIdx = 0;
        this.mode = 'document';
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
        // Remove event listeners
        if (this._handlers.documentKeydown) {
            document.removeEventListener('keydown', this._handlers.documentKeydown);
        }
    }

    toggleUiVisibility() {
        // Hide video/image/audio specific UI
        document.getElementById('videoContainer')?.style.setProperty('display', 'none');
        document.getElementById('imageContainer')?.style.setProperty('display', 'none');
        document.getElementById('audioContainer')?.style.setProperty('display', 'none');
        document.getElementById('segmentNav')?.style.setProperty('display', 'none');
        document.getElementById('progressBar')?.style.setProperty('display', 'none');
        
        // Hide all nav buttons
        ['prevFrame', 'nextFrame', 'playPause', 'prevVideo', 'nextVideo', 'prevImage', 'nextImage', 'prevAudio', 'nextAudio'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        
        // Show document container
        let docContainer = document.getElementById('documentContainer');
        if (!docContainer) {
            // Create document container if it doesn't exist
            const viewer = document.getElementById('viewer');
            const videoContainer = document.getElementById('videoContainer');
            if (viewer && videoContainer) {
                docContainer = document.createElement('div');
                docContainer.id = 'documentContainer';
                docContainer.style.cssText = 'display: flex; flex-direction: column; gap: 15px; padding: 20px; background: #1a202c; border-radius: 8px; min-height: 400px;';
                docContainer.innerHTML = `
                    <div id="documentMeta" style="
                        font-size: 0.9rem;
                        padding: 12px;
                        background: #2d3748;
                        border-radius: 6px;
                        border-left: 4px solid #48bb78;
                    "></div>
                    <div id="documentViewToggle" style="display: flex; gap: 10px; margin-bottom: 10px;">
                        <button id="showSplitBtn" class="view-toggle-btn active" style="
                            padding: 8px 16px;
                            background: #4299e1;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">⬚ Split</button>
                        <button id="showPdfBtn" class="view-toggle-btn" style="
                            padding: 8px 16px;
                            background: #2d3748;
                            color: #a0aec0;
                            border: 1px solid #4a5568;
                            border-radius: 4px;
                            cursor: pointer;
                        ">📄 PDF</button>
                        <button id="showTextBtn" class="view-toggle-btn" style="
                            padding: 8px 16px;
                            background: #2d3748;
                            color: #a0aec0;
                            border: 1px solid #4a5568;
                            border-radius: 4px;
                            cursor: pointer;
                        ">📝 Text</button>
                    </div>
                    <div id="documentSplitContainer" style="display: flex; gap: 15px;">
                        <iframe id="documentPdfSplit" style="
                            flex: 1;
                            min-width: 0;
                            height: 600px;
                            border: none;
                            border-radius: 6px;
                            background: #2d3748;
                        "></iframe>
                        <div id="documentTextSplit" style="
                            flex: 1;
                            min-width: 0;
                            font-size: 0.9rem;
                            line-height: 1.6;
                            padding: 15px;
                            background: #2d3748;
                            border-radius: 6px;
                            height: 600px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                            font-family: 'Segoe UI', system-ui, sans-serif;
                        "></div>
                    </div>
                    <iframe id="documentPdf" style="
                        display: none;
                        width: 100%;
                        height: 600px;
                        border: none;
                        border-radius: 6px;
                        background: #2d3748;
                    "></iframe>
                    <div id="documentText" style="
                        display: none;
                        font-size: 0.95rem;
                        line-height: 1.7;
                        padding: 20px;
                        background: #2d3748;
                        border-radius: 6px;
                        max-height: 600px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                        font-family: 'Segoe UI', system-ui, sans-serif;
                    "></div>
                `;
                viewer.insertBefore(docContainer, videoContainer);
                
                const setActiveBtn = (activeId) => {
                    ['showPdfBtn', 'showTextBtn', 'showSplitBtn'].forEach(id => {
                        const btn = document.getElementById(id);
                        if (btn) {
                            if (id === activeId) {
                                btn.style.background = '#4299e1';
                                btn.style.color = 'white';
                                btn.style.border = 'none';
                            } else {
                                btn.style.background = '#2d3748';
                                btn.style.color = '#a0aec0';
                                btn.style.border = '1px solid #4a5568';
                            }
                        }
                    });
                };
                
                // Add toggle handlers
                document.getElementById('showPdfBtn')?.addEventListener('click', () => {
                    document.getElementById('documentPdf').style.display = 'block';
                    document.getElementById('documentText').style.display = 'none';
                    document.getElementById('documentSplitContainer').style.display = 'none';
                    setActiveBtn('showPdfBtn');
                });
                document.getElementById('showTextBtn')?.addEventListener('click', () => {
                    document.getElementById('documentPdf').style.display = 'none';
                    document.getElementById('documentText').style.display = 'block';
                    document.getElementById('documentSplitContainer').style.display = 'none';
                    setActiveBtn('showTextBtn');
                });
                document.getElementById('showSplitBtn')?.addEventListener('click', () => {
                    document.getElementById('documentPdf').style.display = 'none';
                    document.getElementById('documentText').style.display = 'none';
                    document.getElementById('documentSplitContainer').style.display = 'flex';
                    setActiveBtn('showSplitBtn');
                });
            }
        }
        if (docContainer) {
            docContainer.style.display = 'flex';
        }
        
        // Show quality metrics area for document info
        document.getElementById('qualityMetrics')?.style.setProperty('display', 'block');
        // Hide transcription display
        document.getElementById('transcriptionText')?.style.setProperty('display', 'none');
    }

    buildItemList() {
        const videoList = document.getElementById('videoList');
        if (!videoList) return;
        
        videoList.innerHTML = '';
        this.items.forEach((item, i) => {
            const div = document.createElement('div');
            div.className = 'video-item' + (i === 0 ? ' active' : '');
            // Extract name from annotation path
            const name = item.annotation_path?.split('/').pop()?.replace('.annotation.json', '') || `Document ${i + 1}`;
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
        
        const docMetaEl = document.getElementById('documentMeta');
        const docTextEl = document.getElementById('documentText');
        const titleEl = document.getElementById('videoTitle');
        const metaEl = document.getElementById('videoMeta');
        const qualityEl = document.getElementById('qualityMetrics');
        
        // Update title
        const name = item.annotation_path?.split('/').pop()?.replace('.annotation.json', '') || `Document ${index + 1}`;
        if (titleEl) titleEl.textContent = `Document ${index + 1}/${this.items.length}: ${name}`;

        // Load annotation metadata
        let annotation = null;
        if (item.annotation_path) {
            try {
                const response = await fetch(item.annotation_path, { cache: 'no-store' });
                if (response.ok) {
                    annotation = await response.json();
                }
            } catch (err) {
                console.warn('[DOCUMENT] Error loading annotation:', err);
            }
        }

        // Update meta info
        if (metaEl && annotation) {
            const parts = [];
            if (annotation.source_file) parts.push(annotation.source_file);
            if (annotation.page_count) parts.push(`${annotation.page_count} pages`);
            if (annotation.word_count) parts.push(`${annotation.word_count.toLocaleString()} words`);
            metaEl.textContent = parts.join(' | ') || '-';
        }

        // Update document metadata panel
        if (docMetaEl && annotation) {
            let html = '';
            if (annotation.extraction_method) {
                html += `<div><strong>Extraction:</strong> ${annotation.extraction_method}</div>`;
            }
            if (annotation.char_count) {
                html += `<div><strong>Characters:</strong> ${annotation.char_count.toLocaleString()}</div>`;
            }
            if (annotation.processed_at) {
                const date = new Date(annotation.processed_at).toLocaleString();
                html += `<div><strong>Processed:</strong> ${date}</div>`;
            }
            docMetaEl.innerHTML = html || '<em>No metadata available</em>';
        }

        // Update quality/source info
        if (qualityEl && annotation) {
            qualityEl.innerHTML = '';
            if (annotation.source_file) {
                const line = document.createElement('div');
                line.innerHTML = `<strong>Source:</strong> ${annotation.source_file}`;
                qualityEl.appendChild(line);
            }
            if (annotation.uuid) {
                const line = document.createElement('div');
                line.innerHTML = `<strong>UUID:</strong> <code style="font-size: 0.8em">${annotation.uuid}</code>`;
                qualityEl.appendChild(line);
            }
        }

        // Load PDF in iframe (don't change view mode - just load data)
        const docPdfEl = document.getElementById('documentPdf');
        const docPdfSplitEl = document.getElementById('documentPdfSplit');
        const toggleEl = document.getElementById('documentViewToggle');
        if (docPdfEl) {
            if (item.pdf_path) {
                docPdfEl.src = item.pdf_path;
                if (docPdfSplitEl) docPdfSplitEl.src = item.pdf_path;
                if (toggleEl) toggleEl.style.display = 'flex';
                console.log('[DOCUMENT] Loading PDF:', item.pdf_path);
            } else {
                docPdfEl.src = '';
                if (docPdfSplitEl) docPdfSplitEl.src = '';
                // No PDF available, force text-only view
                docPdfEl.style.display = 'none';
                document.getElementById('documentSplitContainer').style.display = 'none';
                document.getElementById('documentText').style.display = 'block';
                if (toggleEl) toggleEl.style.display = 'none';
            }
        }

        // Load text content
        const docTextSplitEl = document.getElementById('documentTextSplit');
        if (docTextEl && item.text_path) {
            try {
                const response = await fetch(item.text_path, { cache: 'no-store' });
                if (response.ok) {
                    const text = await response.text();
                    // Show first ~10000 chars with truncation notice
                    const maxLen = 10000;
                    let displayText;
                    if (text.length > maxLen) {
                        displayText = text.slice(0, maxLen) + '\n\n... [Truncated - ' + (text.length - maxLen).toLocaleString() + ' more characters]';
                    } else {
                        displayText = text;
                    }
                    docTextEl.textContent = displayText;
                    if (docTextSplitEl) docTextSplitEl.textContent = displayText;
                } else {
                    docTextEl.textContent = '(Failed to load document text)';
                    if (docTextSplitEl) docTextSplitEl.textContent = '(Failed to load document text)';
                }
            } catch (err) {
                console.error('[DOCUMENT] Error loading text:', err);
                docTextEl.textContent = '(Error loading document text)';
                if (docTextSplitEl) docTextSplitEl.textContent = '(Error loading document text)';
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
