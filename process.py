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

INPUT_DIR = Path(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input")
OUTPUT_DIR = Path(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input\Buch test Output")


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_images(input_dir: Path) -> List[Path]:
    """Load image paths from the input directory."""
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return []

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    logging.info("Found %d image(s) in %s", len(image_paths), input_dir)
    return image_paths


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale and apply denoising blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _estimate_background_lab(image: np.ndarray) -> np.ndarray:
    """Estimate background color from the outer image border in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    bw = max(5, int(min(h, w) * 0.04))

    border_pixels = np.concatenate(
        [
            lab[:bw, :, :].reshape(-1, 3),
            lab[-bw:, :, :].reshape(-1, 3),
            lab[:, :bw, :].reshape(-1, 3),
            lab[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def build_page_mask(image: np.ndarray) -> np.ndarray:
    """Build a mask where the book/page region is white and background is black."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg = _estimate_background_lab(image)

    # Distance to border background color in LAB space.
    diff = lab.astype(np.float32) - bg.astype(np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # Keep the largest connected component as book/object region.
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def detect_page_boxes(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect candidate page boxes sorted left-to-right."""
    mask = build_page_mask(image)
    h, w = mask.shape[:2]

    # Try to split the book mask into left/right components at detected seam.
    seam_x, seam_conf = find_split_line(image)
    boxes: List[Tuple[int, int, int, int]] = []

    if seam_conf > 0.08 and int(w * 0.2) < seam_x < int(w * 0.8):
        left_mask = mask.copy()
        right_mask = mask.copy()
        left_mask[:, seam_x + 2 :] = 0
        right_mask[:, : seam_x - 2] = 0

        for side_mask in (left_mask, right_mask):
            contours, _ = cv2.findContours(side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < h * w * 0.05:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            px = max(8, int(bw * 0.02))
            py = max(8, int(bh * 0.02))
            x0, y0 = max(0, x - px), max(0, y - py)
            x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes.sort(key=lambda b: b[0])
    return boxes[:2]


def find_split_line(image: np.ndarray) -> Tuple[int, float]:
    """Estimate center gutter x-position and confidence using robust column statistics."""
    gray = preprocess_image(image)
    h, w = gray.shape[:2]

    center = w // 2
    window = max(20, w // 5)
    start = max(0, center - window)
    end = min(w, center + window)

    region = gray[:, start:end]
    # Robust per-column brightness and edge measures (less sensitive to big illustrations).
    q70 = np.percentile(region, 70, axis=0)
    q90 = np.percentile(region, 90, axis=0)

    grad_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    edge_strength = np.percentile(np.abs(grad_x), 70, axis=0)

    # Gutter tends to be slightly darker and has clear edge transitions.
    darkness = (q90.max() - q70).astype(np.float32)
    score = darkness + 0.8 * edge_strength

    local_idx = int(np.argmax(score))
    split_x = start + local_idx

    center_band = score[max(0, local_idx - 20) : min(score.size, local_idx + 21)]
    baseline = float(np.median(score))
    confidence = float((score[local_idx] - baseline) / (np.std(center_band) + 1e-6))
    confidence = max(0.0, min(confidence / 8.0, 1.0))
    return split_x, confidence


def is_double_page(image: np.ndarray) -> bool:
    """Detect whether the image is likely a double-page spread."""
    h, w = image.shape[:2]
    if w / max(h, 1) < 1.18:
        return False

    seam_x, seam_conf = find_split_line(image)
    has_center_seam = seam_conf > 0.10 and int(w * 0.22) < seam_x < int(w * 0.78)

    boxes = detect_page_boxes(image)
    has_two_boxes = len(boxes) == 2

    return has_center_seam or has_two_boxes


def split_double_page(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a double-page image into left and right pages with overlap safety."""
    h, w = image.shape[:2]
    seam_x, _ = find_split_line(image)
    seam_x = int(np.clip(seam_x, int(w * 0.25), int(w * 0.75)))

    # Small overlap prevents cutting text exactly on gutter.
    overlap = max(8, int(w * 0.01))
    left = image[:, : min(w, seam_x + overlap)]
    right = image[:, max(0, seam_x - overlap) :]

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
    """Find a robust page contour and return a quadrilateral if possible."""
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = h * w * 0.25
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

        x, y, bw, bh = cv2.boundingRect(contour)
        return np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)

    return None


def crop_page(image: np.ndarray) -> np.ndarray:
    """Crop image to the primary page region while preserving borders."""
    mask = build_page_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    px = max(12, int(w * 0.02))
    py = max(12, int(h * 0.02))

    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(image.shape[1], x + w + px)
    y1 = min(image.shape[0], y + h + py)

    return image[y0:y1, x0:x1]


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Apply perspective correction using detected page corners."""
    gray = preprocess_image(image)
    page = find_page_contour(gray)
    if page is None:
        return image

    rect = order_points(page)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    if max_width < 200 or max_height < 200:
        return image

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def dewarp_page(image: np.ndarray) -> np.ndarray:
    """Apply mild vertical dewarping to flatten book curvature near the gutter."""
    h, w = image.shape[:2]
    if h < 300 or w < 300:
        return image

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    x_norm = (x_coords - (w / 2.0)) / (w / 2.0)
    strength = 0.02

    y_offset = (x_norm**2) * strength * h
    map_x = x_coords
    map_y = np.clip(y_coords - y_offset, 0, h - 1)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Normalize lighting while preserving color for illustrations/photos."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    illum = cv2.GaussianBlur(l, (0, 0), sigmaX=31)
    l_corr = cv2.divide(l, illum, scale=220)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_corr = clahe.apply(l_corr)

    merged = cv2.merge((l_corr, a, b))
    color_out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    color_out = cv2.bilateralFilter(color_out, d=5, sigmaColor=30, sigmaSpace=30)

    return color_out


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

            pages = [image]
            if is_double_page(image):
                pages = list(split_double_page(image))

            for page in pages:
                processed = process_page(page)
                if processed.size == 0:
                    logging.warning("Skipping empty result for %s", image_path)
                    continue
                save_page(processed, OUTPUT_DIR, page_index)
                page_index += 1

        except Exception as exc:
            logging.exception("Error processing %s: %s", image_path, exc)


if __name__ == "__main__":
    main()