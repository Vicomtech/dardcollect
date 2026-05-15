/**
 * Viewer Configuration and Constants
 * Shared across image and video viewers
 */

const VIEWER_CONFIG = {
    // Visuals
    COLORS: [
        '#00ff88', '#00d9ff', '#ff0055', '#ffcc00',
        '#cc00ff', '#ff6600', '#00ffcc', '#ff3333'
    ],
    LINE_WIDTH_BBOX: 3,
    LINE_WIDTH_SKELETON: 2,
    KEYPOINT_RADIUS: 3,

    // Text
    FONT: "14px Segoe UI",
    TEXT_BG_PADDING: 10,
    TEXT_OFFSET_X: 5,
    TEXT_OFFSET_Y: 5,

    // Skeleton connections (COCO-Wholebody format)
    SKELETON: [
        // Body (COCO 17 keypoints 0-16)
        [1, 3], [1, 0], [2, 0], [2, 4],   // ears-eyes-nose
        [0, 5], [0, 6],                     // nose to shoulders
        [5, 6],                             // shoulders
        [5, 7], [7, 9],                     // left arm
        [6, 8], [8, 10],                    // right arm
        [5, 11], [6, 12],                   // torso
        [11, 12],                           // hips
        [11, 13], [13, 15],                 // left leg
        [12, 14], [14, 16],                 // right leg
        // Feet (17-22)
        [15, 17], [15, 18], [15, 19],
        [16, 20], [16, 21], [16, 22],
        // Hands (91-132)
        [91, 92], [92, 93], [93, 94], [94, 95],
        [91, 96], [96, 97], [97, 98], [98, 99],
        [91, 100], [100, 101], [101, 102], [102, 103],
        [91, 104], [104, 105], [105, 106], [106, 107],
        [91, 108], [108, 109], [109, 110], [110, 111],
        [112, 113], [113, 114], [114, 115], [115, 116],
        [112, 117], [117, 118], [118, 119], [119, 120],
        [112, 121], [121, 122], [122, 123], [123, 124],
        [112, 125], [125, 126], [126, 127], [127, 128],
        [112, 129], [129, 130], [130, 131], [131, 132],
        // Face (23-90)
        [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
        [31, 32], [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39],
        [40, 41], [41, 42], [42, 43], [43, 44],
        [45, 46], [46, 47], [47, 48], [48, 49],
        [50, 51], [51, 52], [52, 53], [53, 54],
        [55, 56], [56, 57], [57, 58], [58, 59],
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [60, 64],
        [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [73, 74], [74, 75], [68, 72],
        [76, 77], [77, 78], [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [76, 82],
    ]
};

// State management
const VIEWER_STATE = {
    imageMode: false,
    imageDetections: [],
    currentImageIdx: 0,
    currentImageData: null,
    
    videoDetections: [],
    videoFiles: {},
    qualityMap: {},
    transcriptionMap: {},
    currentVideoIndex: 0,
    currentSegmentIndex: 0,
    fps: 30,
    
    isLoading: false,
    savedState: null
};

// LocalStorage keys
const STORAGE_KEY = 'viewer_state';

// Utility function to get color for ID
function getColorForId(id) {
    return VIEWER_CONFIG.COLORS[id % VIEWER_CONFIG.COLORS.length];
}

// Format time in MM:SS
function formatTime(seconds) {
    if (seconds == null) return '?';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}
