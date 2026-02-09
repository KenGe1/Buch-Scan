"""
Batch process photos of book pages into high-quality scanned pages.

Usage:
    python process.py

Requirements:
    - Place input images (JPG/PNG) in the "input/" folder.
    - Processed pages are saved to the "output/" folder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_images(input_dir: Path) -> List[Path]:
    """Load image paths from the input directory."""
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return []
    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    logging.info("Found %d image(s) in %s", len(image_paths), input_dir)
    return image_paths


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply a mild blur for noise reduction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def is_double_page(image: np.ndarray) -> bool:
    """Detect whether the image is likely a double-page spread."""
    h, w = image.shape[:2]
    if w / max(h, 1) < 1.4:
        return False

    gray = preprocess_image(image)
    projection = gray.mean(axis=0)
    center = w // 2
    window = w // 6
    start = max(center - window, 0)
    end = min(center + window, w)
    center_slice = projection[start:end]

    if center_slice.size == 0:
        return False

    center_valley = center_slice.min()
    overall_mean = projection.mean()

    valley_ratio = center_valley / max(overall_mean, 1.0)
    logging.debug("Double-page valley ratio: %.3f", valley_ratio)
    return valley_ratio < 0.7


def split_double_page(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a double-page image into left and right halves."""
    gray = preprocess_image(image)
    projection = gray.mean(axis=0)
    w = image.shape[1]
    center = w // 2
    window = w // 6
    start = max(center - window, 0)
    end = min(center + window, w)

    valley_index = int(start + np.argmin(projection[start:end])) if end > start else center
    split_col = int(np.clip(valley_index, w * 0.4, w * 0.6))

    left = image[:, :split_col]
    right = image[:, split_col:]
    return left, right


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_page_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest contour that approximates a quadrilateral page."""
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def crop_page(image: np.ndarray) -> np.ndarray:
    """Crop image to the detected page contour."""
    gray = preprocess_image(image)
    page = find_page_contour(gray)
    if page is None:
        logging.warning("No page contour found; using original image.")
        return image

    rect = order_points(page)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Apply perspective correction using detected page corners."""
    gray = preprocess_image(image)
    page = find_page_contour(gray)
    if page is None:
        logging.warning("No perspective correction applied; using original image.")
        return image

    rect = order_points(page)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def dewarp_page(image: np.ndarray) -> np.ndarray:
    """Apply a gentle dewarping to reduce curved page distortion."""
    h, w = image.shape[:2]
    k1 = -0.0006
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    camera_matrix = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32
    )
    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)
    new_camera, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Normalize lighting and shadows using adaptive histogram equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=31)
    corrected = cv2.divide(gray, blur, scale=255)
    corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

    return corrected_bgr


def save_page(image: np.ndarray, output_dir: Path, index: int) -> None:
    """Save the processed page image with sequential naming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"page_{index:04d}.jpg"
    cv2.imwrite(str(filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    logging.info("Saved %s", filename)


def process_page(image: np.ndarray) -> np.ndarray:
    """Process a single page through crop, perspective correction, dewarp, and lighting."""
    cropped = crop_page(image)
    corrected = correct_perspective(cropped)
    dewarped = dewarp_page(corrected)
    normalized = normalize_lighting(dewarped)
    return normalized


def main() -> None:
    """Entry point for batch processing."""
    image_paths = load_images(INPUT_DIR)
    if not image_paths:
        logging.error("No images to process.")
        return

    page_index = 1
    for image_path in image_paths:
        logging.info("Processing %s", image_path)
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.error("Failed to read image: %s", image_path)
                continue

            if is_double_page(image):
                left, right = split_double_page(image)
                for page in (left, right):
                    processed = process_page(page)
                    save_page(processed, OUTPUT_DIR, page_index)
                    page_index += 1
            else:
                processed = process_page(image)
                save_page(processed, OUTPUT_DIR, page_index)
                page_index += 1
        except Exception as exc:
            logging.exception("Error processing %s: %s", image_path, exc)


if __name__ == "__main__":
    main()
