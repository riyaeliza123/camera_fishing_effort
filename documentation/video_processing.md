# From Images to Videos: Processing Extension Guide

## Introduction

The current application processes **static images** using YOLOv8. Extending this to handle **videos** is fundamentally straightforward: a video is simply a **continuous sequence of image frames** played at a specific frame rate (e.g., 30 fps).

Processing videos is **just one architectural step away** from image processing:

```
Current Flow:        Video Extension:
Image → Inference    Video → Extract Frames → Inference → Aggregate Results
                          ↓
                     (This is the new step)
```

This document outlines what's needed to add video support while reusing 90% of existing inference code.

---

## Core Concept: Frame Extraction

### How Videos Work

- **Video File**: Sequence of frames encoded with compression
- **Frame Rate**: Frames per second (fps) - typically 24-60fps
- **Duration**: Total frames = duration (seconds) × fps
- **Challenge**: Processing every frame from a 10-minute video at 30fps = 18,000 frames (intensive!)

### Solution: Frame Sampling

Instead of processing every frame, we **intelligently select a subset** of frames for inference. This reduces computation while maintaining detection coverage.

**Recommended Strategy**: **Interval-based sampling** (every Nth frame)
- Deterministic and consistent across all videos
- Good balance between coverage and speed
- Easy to configure and debug
- Works well for long surveillance videos

---

## Frame Sampling Strategies

### 1. **Random Sampling**
**Extract N randomly-selected frames from the entire video**

```
Example: 10-minute video at 30fps = 18,000 total frames
Strategy: Select 30 random frames
Result: ~0.17% coverage, instant processing
```

**Pros**: Simple, truly representative

**Cons**: May miss isolated events, less predictable

**When to use**: 
- Quick screening if a video contains boats
- Proof-of-concept
- High-volume batch processing where precision matters less

---

### 2. **Interval-Based Sampling (Fixed Step)**
**Extract every Nth frame**

```
Example: Extract every 100th frame
Video (18,000 frames) → Process frames [0, 100, 200, 300, ...]
Result: 180 frames for analysis
```

**Pros**: Deterministic, consistent coverage, easy to calculate

**Cons**: May miss fast-moving boats between intervals

**When to use**:
- Standard batch processing
- Consistent sampling across different videos
- When you need exactly N frames

---

### 3. **Uniform Time Distribution**
**Evenly space frame samples across entire video duration**

```
Example: Divide 10-minute video into 6 segments
Extract 1 frame from start of each segment
Result: 6 equally-spaced frames covering 0%, 16%, 33%, 50%, 66%, 83% of video
```

**Pros**: Balanced temporal coverage, captures different scenes

**Cons**: May still miss patterns

**When to use**:
- Video summarization
- Detecting changes over time (morning → afternoon → evening)
- Scenario analysis

---

### 4. **Adaptive/Smart Sampling**
**Select frames based on motion detection or scene changes**

```
Pseudo-logic:
1. Quick scan identifies frames with motion
2. Prioritize high-motion frames for detailed inference
3. Sample evenly from remaining frames

Result: More frames where action happens, fewer in static scenes
```

**Pros**: Highly efficient, focuses on relevant content

**Cons**: Requires pre-processing pass, more complex

**When to use**:
- Production systems with performance requirements
- Real-time monitoring (use motion to trigger inference)

---

## Integration with Existing Inference Pipeline

### Current Flow (Images)

```python
# User uploads image
image_bytes → stretch to 640×640 → run chokepoint.py inference → annotate → return results
```

### Extended Flow (Videos)

```python
# User uploads video
video_file → extract frames (sampling strategy) → 
    FOR EACH frame:
        stretch to 640×640 → run chokepoint.py inference → annotate → track detections
    → aggregate results (counts, statistics) → return annotated frames + video CSV
```

### Code Reuse

The **inference logic is unchanged**:
- `scripts/chokepoint.py` → `model.predict()` works on both images and extracted frames
- `scripts/fishing.py` → Roboflow API works the same
- `scripts/utils.py` → `annotate_image()` works on every frame
- `scripts/dataframe.py` → Create DataFrame with frame-level results

**New code needed**:
- Frame extraction module (e.g., `scripts/video_processor.py`)
- Sampling strategy implementations
- Result aggregation logic (sum detections across frames)

---

## Processing Modes: Synchronous vs Asynchronous

### Mode A: Synchronous Processing (Simple)

**User uploads → Server processes immediately → Results returned when ready**

```
Workflow:
1. User uploads video file (pre-split to 1-2 hours)
2. Server extracts frames using interval-based sampling (every 100th frame)
3. Inference runs sequentially on selected frames
4. Results returned to user
```

**Pros**:
- Simple implementation (no background job infrastructure)
- Predictable: user knows processing happens now
- Good for 1-2 hour videos (manageable time)

**Cons**:
- Long wait time for user (1-hour video with every 100th frame = ~9 min)
- Blocks worker thread (not suitable for high concurrency)
- Lost connection during processing = lost progress

**Best for**: Small deployments, development/testing

---

### Mode B: Asynchronous Processing (Recommended)

**User uploads → Server returns job ID → User polls for progress → Results ready when done**

```
Workflow:
1. User uploads video (1-2 hours)
2. Server queues job and returns immediately with job_id
3. Background worker process extracts frames and runs inference
4. User can check status: GET /job/{job_id}/status
5. When complete: results available for download
```

**Pros**:
- Non-blocking: user doesn't wait in browser
- Can handle multiple uploads simultaneously  
- Better UX: show real-time progress bar
- Handles network disconnections gracefully
- Scales better as user base grows

**Cons**:
- More complex architecture
- Requires background worker process

**Implementation Options**:
- **Simplest**: FastAPI BackgroundTasks (built-in, no extra dependencies)
- **Better**: File-based job queue with separate worker process
- **Best**: Redis + Celery (if you add many async features later)

**Recommendation for Phase 1**: Use FastAPI BackgroundTasks - lowest friction to add video support

---

### Mode C: Real-Time Streaming (Future Enhancement)

**Live camera feed → Process frames as they arrive → Real-time alerts**

```
Workflow:
1. Camera streams video to app (RTMP, HLS, or WebSocket)
2. Extract frames at configurable interval
3. Run inference with minimal latency
4. Return detections in near real-time
5. Store summary file (CSV) of day's detections
```

**Use case**: Live camera feeds with instant detection alerts


**Why deferred**:
- Requires WebSocket/streaming protocol support (major architecture change)
- Needs low-latency model optimization (likely GPU required)
- Fly.io shared CPU insufficient for 30fps real-time processing
- Revisit after basic video processing works well


### Recommended Approach (Phase 1)

1. **Sampling Strategy**: Interval-based
   - Default: every 100th frame
   - Allow users to configure: 50, 100, 200, 500 (pre-set options)

2. **Processing Mode**: Asynchronous (background job)
   - User uploads → immediately returns job ID
   - Background worker processes frames
   - User checks status via progress endpoint

3. **Output Format**: Annotated frame images
   - Return individual JPG frames with bounding boxes
   - Include CSV summary of all detections
   - User downloads as ZIP archive

**Processing Time Estimates** (1-hour video, Chokepoint model):
- Every 100th frame: 9 min processing → 1,080 frame images → 350MB output
- Every 500th frame: 2 min processing → 216 frame images → 65MB output (quick mode)

---

## Output Format Options Explained

### Option A: Annotated Frame Images (Phase 1 - Recommended) 

**Return selected frame images with bounding boxes drawn**

**Example Output**:
```
results_abc123.zip
├── frames/
│   ├── frame_0000.jpg (with bounding boxes)
│   ├── frame_0100.jpg (with bounding boxes)
│   ├── frame_0200.jpg (with bounding boxes)
│   └── ...
├── results.csv (detection summary)
└── summary.json (aggregate statistics)
```

**Pros**:
- Fast to generate (just draw boxes on extracted frames)
- Lightweight output (~320MB for 1-hour video)
- Users can view and zoom individual frames
- Easy to debug detections
- Low CPU/storage overhead

**Cons**:
- Gaps between frames not visible
- Cannot recreate smooth video playback

**Processing Time**: 9 minutes (1-hour video, every 100th frame)

---

### Option B: Full Output Video with Annotations (Future) 

**Generate complete video file with annotations on every frame**

**Example Output**:
```
results_abc123.zip
├── output_video.mp4 (full video, all frames, 30fps, with boxes)
├── result.csv
└── summary.json
```

**Pros**:
- Complete visual record of entire video
- Smooth playback shows motion

**Cons**:
- Very resource-intensive (must re-encode entire video)
- Huge output files (2-4GB for 1-hour video)
- Slow (adds 50-100% to processing time)
- High storage and bandwidth costs

**Processing Time**: 30-40 minutes (1-hour video, re-encoding all frames)

---


## Architecture Summary

**Video Processing Pipeline**:

```
User Upload (MP4, 1-2 hr)
        ↓
    /upload endpoint
        ↓
        ├─→ Auto-detect: is it video? → Yes
        ├─→ Queue job (return job_id immediately)
        ├─→ Background worker starts
        │
        └─→ Video Job Processing (async)
            ├─→ Extract frames (every 100th)
            ├─→ FOR EACH frame:
            │   ├─→ Resize to 640×640
            │   ├─→ Run inference (chokepoint or fishing)
            │   ├─→ Save annotated JPEG
            │   └─→ Update progress
            ├─→ Create results.csv
            ├─→ Generate summary.json
            ├─→ Package into ZIP
            └─→ Mark job as completed

User Actions (Polling):
├─→ GET /job/{job_id} → Check progress
├─→ Wait for status = "completed"
└─→ GET /download/{job_id} → Download ZIP
```

