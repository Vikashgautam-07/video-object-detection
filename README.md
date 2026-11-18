# Video Object Detection

Detect objects in webcam or video frames using OpenCV's SSD MobileNet model.
The program draws bounding boxes, shows the detected classes and counts for
each frame, and can save those per-frame counts to a CSV file.

## Project structure

```text
video-object-detection/
├── main.py                 # Command-line entry point
├── detector.py             # Loads the model, detects objects, draws boxes
├── counter.py              # Counts objects by class and draws counts
├── requirements.txt
├── models/
│   ├── frozen_inference_graph.pb
│   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
│   └── coco.names
├── sample_video/           # Example videos
├── img/                    # Project images
└── notebooks/
    └── phase1_exploration.ipynb
```

## Setup

Python 3 and `venv` are recommended. On Ubuntu/WSL, install venv support if
needed with `sudo apt install python3-venv`.

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

When the virtual environment is active, run the program with `python`.

## Run

Use the default webcam (camera index `0`):

```bash
python main.py
```

Use a video file:

```bash
python main.py sample_video/classroom.mp4
```

Change the minimum confidence threshold (default is `0.5`):

```bash
python main.py sample_video/classroom.mp4 --thres 0.65
```

Run for a limited time, set the capture size, or avoid opening a display
window:

```bash
python main.py sample_video/classroom.mp4 --seconds 30 --width 640 --height 480
python main.py sample_video/classroom.mp4 --no-window
```

Press `q` in the video window to stop early. By default, the program processes
the video until it ends.

## Save counts to CSV

Save per-frame class counts to a named file:

```bash
python main.py sample_video/classroom.mp4 --save-csv results.csv
```

If no filename is provided, the output is saved as `detections.csv`:

```bash
python main.py sample_video/classroom.mp4 --save-csv
```

The CSV contains a frame number, detected class, and count for that frame.
Frames with no detections are recorded with a count of zero.

## Options

Run `python main.py --help` to see all options:

- `source`: Webcam index or video path (default: `0`)
- `--thres`: Minimum confidence from `0` to `1` (default: `0.5`)
- `--seconds`: Stop after the specified number of seconds; `0` means no time
  limit
- `--width`, `--height`: Requested capture size (default: `640x480`)
- `--save-csv [FILE]`: Save per-frame counts; default filename is
  `detections.csv`
- `--no-window`: Process without displaying the video

The terminal's **Final counts** are the counts from the last processed frame,
not totals accumulated over the entire video. Use the CSV for per-frame
results.

## Webcam on WSL

If the program reports that no webcam is available, the current environment
does not have a camera device such as `/dev/video0`. Use a video file, or
configure camera/USB forwarding to WSL before trying the webcam again.

## License

Licensed under the Apache-2.0 License. See [LICENSE](LICENSE).
