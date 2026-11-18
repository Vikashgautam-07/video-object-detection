#!/usr/bin/env python3

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2

from counter import count_detections, draw_counts
from detector import ObjectDetector


PROJECT_FOLDER = Path(__file__).resolve().parent
MODELS_FOLDER = PROJECT_FOLDER / "models"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects and count them by class."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Webcam number or video file path (default: 0)",
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.5,
        help="Minimum confidence threshold, from 0 to 1 (default: 0.5)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after this many seconds; 0 runs until stopped (default)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default: 480)",
    )
    parser.add_argument(
        "--save-csv",
        nargs="?",
        const="detections.csv",
        metavar="FILE",
        help="Save frame counts to CSV (default file: detections.csv)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without displaying the video window",
    )
    return parser.parse_args(argv)


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        capture.release()
        if source.isdigit():
            raise RuntimeError(
                "No webcam is available. This system has no camera device "
                "(/dev/video0). Use a video file instead: "
                "python3 main.py path/to/video.mp4"
            )
        raise RuntimeError(
            "Could not open video file: "
            + source
            + ". Check that the file exists and has a supported format."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    detector = ObjectDetector(
        weights=MODELS_FOLDER / "frozen_inference_graph.pb",
        config=MODELS_FOLDER / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
        labels=MODELS_FOLDER / "coco.names",
        threshold=args.thres,
    )
    capture = open_capture(args.source, args.width, args.height)

    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "label", "count"])

    start_time = time.monotonic()
    frame_count = 0
    detection_time = 0.0
    final_counts: Counter[str] = Counter()

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if args.seconds > 0 and elapsed >= args.seconds:
                print("Time limit reached.")
                break

            has_frame, frame = capture.read()
            if not has_frame:
                print("No more frames or the video could not be read.")
                break

            frame_count += 1
            detection_start = time.monotonic()
            try:
                detections = detector.detect(frame)
            except cv2.error as error:
                print("Detection error:", error)
                return 1
            detection_time += time.monotonic() - detection_start

            frame_counts = count_detections(detections)
            final_counts = frame_counts
            frame = detector.draw(frame, detections)
            draw_counts(frame, frame_counts)

            if csv_writer:
                if frame_counts:
                    for label, count in sorted(frame_counts.items()):
                        csv_writer.writerow([frame_count, label, count])
                else:
                    csv_writer.writerow([frame_count, "", 0])

            if not args.no_window:
                cv2.imshow("Video Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

    total_time = time.monotonic() - start_time
    print("Final counts:")
    if final_counts:
        for label, count in sorted(final_counts.items()):
            print(f"{label}: {count}")
    else:
        print("No objects detected")

    if frame_count:
        print(f"Frames processed: {frame_count}")
        print(f"FPS: {frame_count / total_time:.2f}")
        print(f"Average detection time: {detection_time / frame_count:.4f} seconds")

    if args.save_csv:
        print("CSV saved to:", args.save_csv)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print("Error:", error)
        raise SystemExit(1)
