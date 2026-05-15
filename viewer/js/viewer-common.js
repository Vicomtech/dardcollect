/**
 * Common functionality shared between image and video viewers
 */

// DOM elements (initialize on first use)
let dropZone, fileInput, viewer, videoList, videoCount, frameInfo, segmentInfo;
let showBBoxCheck, showKeypointsCheck, showScoresCheck, showIdsCheck;
let showFaceCropArcfaceCheck, showFaceCropOfiqCheck, kptThresholdInput;

// Initialize DOM references
function initializeDomElements() {
    dropZone = dropZone || document.getElementById('dropZone');
    fileInput = fileInput || document.getElementById('fileInput');
    viewer = viewer || document.getElementById('viewer');
    videoList = videoList || document.getElementById('videoList');
    videoCount = videoCount || document.getElementById('videoCount');
    frameInfo = frameInfo || document.getElementById('frameInfo');
    segmentInfo = segmentInfo || document.getElementById('segmentInfo');
    
    showBBoxCheck = showBBoxCheck || document.getElementById('showBBox');
    showKeypointsCheck = showKeypointsCheck || document.getElementById('showKeypoints');
    showScoresCheck = showScoresCheck || document.getElementById('showScores');
    showIdsCheck = showIdsCheck || document.getElementById('showIds');
    showFaceCropArcfaceCheck = showFaceCropArcfaceCheck || document.getElementById('showFaceCropArcface');
    showFaceCropOfiqCheck = showFaceCropOfiqCheck || document.getElementById('showFaceCropOfiq');
    kptThresholdInput = kptThresholdInput || document.getElementById('kptThreshold');
}

/**
 * Draw a quadrilateral (face crop corners) with scaled coordinates
 */
function drawScaledQuad(ctx, corners, color, scaleX, scaleY) {
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

/**
 * Draw detections scaled to image
 */
function drawImageDetectionsScaled(ctx, detections, scaleX, scaleY) {
    const kptThreshold = parseFloat(kptThresholdInput?.value) || 0.1;

    detections.forEach((det, personIdx) => {
        const color = getColorForId(personIdx);

        // Draw bbox
        if (det.bbox_tlbr && showBBoxCheck?.checked) {
            const [x1, y1, x2, y2] = det.bbox_tlbr;
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(2, VIEWER_CONFIG.LINE_WIDTH_BBOX * Math.min(scaleX, scaleY));
            ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

            if (showScoresCheck?.checked && det.bbox_confidence) {
                ctx.fillStyle = color;
                ctx.font = Math.max(12, 14 * Math.min(scaleX, scaleY)) + 'px Segoe UI';
                ctx.fillText(`Person ${personIdx + 1}`, (x1 + 5) * scaleX, (y1 - 5) * scaleY);
            }
        }

        // Draw face crop corners
        if (showFaceCropArcfaceCheck?.checked && det.face_crop_corners_arcface) {
            drawScaledQuad(ctx, det.face_crop_corners_arcface, '#ffcc00', scaleX, scaleY);
        }
        if (showFaceCropOfiqCheck?.checked && det.face_crop_corners_ofiq) {
            drawScaledQuad(ctx, det.face_crop_corners_ofiq, '#cc66ff', scaleX, scaleY);
        }

        // Draw keypoints and skeleton
        if (det.keypoints && det.keypoint_scores) {
            const kpts = det.keypoints;
            const scores = det.keypoint_scores;

            console.log(`[DEBUG] Drawing detections for person ${personIdx}: kpts=${kpts.length}, scores=${scores.length}, threshold=${kptThreshold}, showKeypoints=${showKeypointsCheck?.checked}`);

            // Draw skeleton connections first
            if (showKeypointsCheck?.checked && VIEWER_CONFIG.SKELETON) {
                ctx.strokeStyle = color;
                // Use a fixed line width rather than scaled, for better visibility
                ctx.lineWidth = 2;
                let skeletonDrawn = 0;
                VIEWER_CONFIG.SKELETON.forEach(([i, j]) => {
                    if (i < kpts.length && j < kpts.length && 
                        kpts[i] && kpts[j] &&
                        scores[i] > kptThreshold && scores[j] > kptThreshold) {
                        ctx.beginPath();
                        ctx.moveTo(kpts[i][0] * scaleX, kpts[i][1] * scaleY);
                        ctx.lineTo(kpts[j][0] * scaleX, kpts[j][1] * scaleY);
                        ctx.stroke();
                        skeletonDrawn++;
                    }
                });
            }

            // Draw keypoints as circles
            if (showKeypointsCheck?.checked) {
                ctx.fillStyle = color;
                // Use a minimum radius for visibility on scaled images
                const kptRadius = Math.max(3, VIEWER_CONFIG.KEYPOINT_RADIUS * Math.min(scaleX, scaleY));
                kpts.forEach((kpt, i) => {
                    if (kpt && Array.isArray(kpt) && kpt.length >= 2 && scores[i] > kptThreshold) {
                        try {
                            ctx.beginPath();
                            ctx.arc(kpt[0] * scaleX, kpt[1] * scaleY, kptRadius, 0, 2 * Math.PI);
                            ctx.fill();
                        } catch (e) {
                            // Skip invalid keypoint
                        }
                    }
                });
            }
        }
    });
}

/**
 * Fetch and load from server data_index.json
 */
async function loadFromServer() {
    try {
        const response = await fetch('data_index.json');
        if (!response.ok) {
            console.log("No data_index.json found");
            return null;
        }
        return await response.json();
    } catch (e) {
        console.log("Error loading server data:", e);
        return null;
    }
}

/**
 * Show/hide UI elements based on mode
 */
function toggleUiVisibility(mode) {
    const videoContainer = document.getElementById('videoContainer');
    const imageContainer = document.getElementById('imageContainer');
    const progressBar = document.getElementById('progressBar');
    const segmentNav = document.getElementById('segmentNav');
    
    if (mode === 'image') {
        // Show image container, hide video
        if (videoContainer) videoContainer.style.display = 'none';
        if (imageContainer) imageContainer.style.display = 'flex';
        if (progressBar) progressBar.style.display = 'none';
        if (segmentNav) segmentNav.style.display = 'none';
        
        // Hide video buttons
        ['prevVideo', 'prevFrame', 'playPause', 'nextFrame', 'nextVideo'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.style.display = 'none';
        });
        // Show image buttons
        ['prevImage', 'nextImage'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.style.display = 'inline-block';
        });
    } else if (mode === 'video') {
        // Show video container, hide image
        if (videoContainer) videoContainer.style.display = 'flex';
        if (imageContainer) imageContainer.style.display = 'none';
        if (progressBar) progressBar.style.display = 'block';
        if (segmentNav) segmentNav.style.display = 'flex';
        
        // Show video buttons
        ['prevVideo', 'prevFrame', 'playPause', 'nextFrame', 'nextVideo'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.style.display = 'inline-block';
        });
        // Hide image buttons
        ['prevImage', 'nextImage'].forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.style.display = 'none';
        });
    }
}

/**
 * Show viewer and hide drop zone
 */
function showViewer() {
    if (dropZone) dropZone.style.display = 'none';
    if (viewer) viewer.classList.add('active');
}

/**
 * Hide viewer and show drop zone
 */
function showDropZone() {
    if (dropZone) dropZone.style.display = 'block';
    if (viewer) viewer.classList.remove('active');
}

/**
 * Score color based on value
 */
function scoreColor(v) {
    return v >= 70 ? '#00ff88' : v >= 40 ? '#ffcc00' : '#ff4444';
}

/**
 * LocalStorage helpers
 */
function saveState(data) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (e) {}
}

function getSavedState() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY));
    } catch (e) {
        return null;
    }
}

// Export for use in other modules
window.ViewerCommon = {
    initializeDomElements,
    drawScaledQuad,
    drawImageDetectionsScaled,
    loadFromServer,
    toggleUiVisibility,
    showViewer,
    showDropZone,
    scoreColor,
    saveState,
    getSavedState
};
