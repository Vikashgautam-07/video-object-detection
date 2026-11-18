from collections import Counter

from detector import Detection


def count_detections(detections: list[Detection]) -> Counter[str]:
    return Counter(detection.label for detection in detections)


def draw_counts(frame, counts: Counter[str]) -> None:
    import cv2

    total = sum(counts.values())
    cv2.putText(
        frame,
        f"Objects: {total}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    y_position = 85
    for label, count in sorted(counts.items()):
        cv2.putText(
            frame,
            f"{label}: {count}",
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        y_position += 30
