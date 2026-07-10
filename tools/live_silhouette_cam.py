import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gait.config import ExperimentConfig
from gait.preprocessing import clean_and_align
from opshora_archive.phase1_hamilton import run_hamilton_pipeline


MODEL_PATH = "yolo26m-seg.pt"
CAMERA_INDEX = 0


def get_person_silhouette(frame, result):
    """Return a binary silhouette: human = white, background = black."""
    height, width = frame.shape[:2]
    silhouette = np.zeros((height, width), dtype=np.uint8)

    if result.masks is None:
        return silhouette

    for mask_tensor in result.masks.data:
        mask = mask_tensor.cpu().numpy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8) * 255
        silhouette = cv2.bitwise_or(silhouette, mask)

    return silhouette


def get_hamilton_skeleton_panel(silhouette):
    """Create a 64x64 aligned silhouette and Hamilton skeleton preview."""
    config = ExperimentConfig(height=64, width=64)
    aligned = clean_and_align(silhouette > 0, config).astype(np.uint8) * 255

    if aligned.sum() == 0:
        skeleton = np.zeros_like(aligned)
    else:
        result = run_hamilton_pipeline(aligned)
        skeleton = result["hamilton_skeleton"].astype(np.uint8) * 255

    aligned_big = cv2.resize(aligned, (320, 320), interpolation=cv2.INTER_NEAREST)
    skeleton_big = cv2.resize(skeleton, (320, 320), interpolation=cv2.INTER_NEAREST)
    return aligned_big, skeleton_big


def add_title(image, title):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Live silhouette + skeleton test started.")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame empty or disconnected.")
            break

        result = model(frame, classes=[0], verbose=False)[0]
        silhouette = get_person_silhouette(frame, result)
        aligned, skeleton = get_hamilton_skeleton_panel(silhouette)

        frame_small = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
        silhouette_small = cv2.resize(silhouette, (320, 320), interpolation=cv2.INTER_NEAREST)

        panels = [
            add_title(frame_small, "camera"),
            add_title(silhouette_small, "silhouette"),
            add_title(aligned, "aligned 64x64"),
            add_title(skeleton, "Hamilton skeleton"),
        ]
        display = np.concatenate(panels, axis=1)
        cv2.imshow("Silhouette and Skeleton Test", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
