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
from PIL import Image

INPUT_DIR = Path(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input")
OUTPUT_DIR = Path(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input\Buch test Output")
OUTPUT_AS_PDF = False
PDF_FILENAME = "book_scan.pdf"
OUTPUT_FORMAT = "jpg"  # "jpg" or "png"
JPEG_QUALITY = 98
PNG_COMPRESSION = 3

ENABLE_PERSPECTIVE_CORRECTION = True
ENABLE_CROP = True
ENABLE_DEWARP = True
ENABLE_LIGHTING_NORMALIZATION = False


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

    # If we only captured text (very small foreground), invert to avoid text-only crops.
    white_ratio = float(mask.mean() / 255.0)
    if white_ratio < 0.18:
        mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep all cleaned components. The final page selection uses shape scoring,
    # which is more robust than always taking the largest blob.
    return mask


def _page_shape_score(contour: np.ndarray, image_h: int, image_w: int) -> float:
    """Score how likely a contour is a full page/book region (higher is better)."""
    area = cv2.contourArea(contour)
    if area <= 1:
        return -1.0

    x, y, bw, bh = cv2.boundingRect(contour)
    rect_area = float(max(1, bw * bh))
    area_ratio = area / float(image_h * image_w)
    fill_ratio = area / rect_area
    aspect = bw / float(max(1, bh))

    # Page-like contours should have significant area, be reasonably rectangular,
    # and usually extend low in the image (useful against chapter-heading/text blobs).
    aspect_score = max(0.0, 1.0 - min(abs(aspect - 0.75), abs(aspect - 1.4)) / 1.4)
    bottom_reach = (y + bh) / float(max(1, image_h))

    return (
        2.6 * area_ratio
        + 0.9 * fill_ratio
        + 0.45 * aspect_score
        + 0.35 * bottom_reach
    )


def _select_best_page_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Select the most page-like contour from the cleaned mask."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = h * w * 0.06
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return None

    return max(candidates, key=lambda c: _page_shape_score(c, h, w))


def detect_book_region(mask: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Return the most likely page/book contour and a padded bounding box."""
    h, w = mask.shape[:2]
    contour = _select_best_page_contour(mask)
    if contour is None:
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    pad_x = max(12, int(bw * 0.025))
    pad_y_top = max(10, int(bh * 0.02))
    pad_y_bottom = max(18, int(bh * 0.055))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y_top)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y_bottom)

    return contour, (x0, y0, x1, y1)


def _rotation_from_contour(contour: np.ndarray) -> float:
    """Estimate small deskew angle (degrees) from the full book contour."""
    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), angle = rect

    if rw < 1 or rh < 1:
        return 0.0

    # OpenCV angle is in [-90, 0): normalize to a small correction around 0.
    if rw < rh:
        angle = angle + 90.0
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90

    return float(np.clip(angle, -20.0, 20.0))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around center while preserving full canvas."""
    if abs(angle) < 0.15:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC)


def align_book_image(image: np.ndarray) -> np.ndarray:
    """Globally align the entire book before split/crop/warp operations."""
    mask = build_page_mask(image)
    region = detect_book_region(mask)
    if region is None:
        return image

    contour, _ = region
    angle = _rotation_from_contour(contour)
    if abs(angle) < 0.35:
        return image

    aligned = rotate_image(image, angle)
    logging.debug("Applied global deskew: %.2f°", angle)
    return aligned


def detect_page_boxes(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect candidate page boxes sorted left-to-right."""
    mask = build_page_mask(image)
    h, w = mask.shape[:2]

    region = detect_book_region(mask)
    if region is None:
        return []

    _, (bx0, by0, bx1, by1) = region

    # Try to split the book mask into left/right components at detected seam.
    seam_x, seam_conf = find_split_line(image)
    boxes: List[Tuple[int, int, int, int]] = []

    if seam_conf > 0.08 and bx0 + int((bx1 - bx0) * 0.2) < seam_x < bx1 - int((bx1 - bx0) * 0.2):
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
            x0, y0 = max(bx0, x - px), max(by0, y - py)
            x1, y1 = min(bx1, x + bw + px), min(by1, y + bh + py)
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes.sort(key=lambda b: b[0])
    return boxes[:2]


def find_split_line(image: np.ndarray) -> Tuple[int, float]:
    """Estimate center gutter x-position and confidence using robust column statistics."""
    gray = preprocess_image(image)
    h, w = gray.shape[:2]

    mask = build_page_mask(image)
    region = detect_book_region(mask)
    if region is not None:
        _, (bx0, _, bx1, _) = region
        center = (bx0 + bx1) // 2
        book_w = max(1, bx1 - bx0)
        window = max(20, int(book_w * 0.28))
    else:
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

    idx = np.arange(score_width := region.shape[1], dtype=np.float32)
    center_bias = 1.0 - np.clip(np.abs(idx - (score_width / 2.0)) / max(1.0, score_width / 2.0), 0, 1)
    # Soft center prior reduces false splits from edge illustrations.
    score = darkness + 0.85 * edge_strength + 0.15 * center_bias * np.max(darkness + 1e-6)

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
    mask = build_page_mask(image)
    region = detect_book_region(mask)
    if region is not None:
        _, (bx0, _, bx1, _) = region
        seam_x = int(np.clip(seam_x, bx0 + int((bx1 - bx0) * 0.2), bx1 - int((bx1 - bx0) * 0.2)))
    else:
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


def find_page_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Find a robust page contour and return a quadrilateral if possible."""
    mask = build_page_mask(image)
    h, w = mask.shape[:2]

    contour = _select_best_page_contour(mask)
    if contour is None:
        return None

    peri = cv2.arcLength(contour, True)
    for eps in (0.015, 0.02, 0.03, 0.04):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def crop_page(image: np.ndarray) -> np.ndarray:
    """Crop image to the primary page region while preserving borders."""
    mask = build_page_mask(image)
    region = detect_book_region(mask)
    if region is None:
        return image

    c, (bx0, by0, bx1, by1) = region
    img_h, img_w = image.shape[:2]
    if cv2.contourArea(c) < img_h * img_w * 0.35:
        return image

    x, y, w, h = cv2.boundingRect(c)
    px = max(12, int(w * 0.02))
    py_top = max(12, int(h * 0.02))
    py_bottom = max(18, int(h * 0.07))

    contour_bottom = int(np.percentile(c[:, 0, 1], 98))
    x0 = max(bx0, x - px)
    y0 = max(by0, y - py_top)
    x1 = min(bx1, x + w + px)
    y1 = min(by1, max(y + h + py_bottom, contour_bottom + py_bottom))

    # Bottom safety: avoid common under-crop on the page footline.
    min_bottom = by0 + int((by1 - by0) * 0.96)
    y1 = max(y1, min(by1, min_bottom))

    return image[y0:y1, x0:x1]


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Apply perspective correction using detected page corners."""
    page = find_page_contour(image)
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
    return cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)


def dewarp_page(image: np.ndarray) -> np.ndarray:
    """Apply adaptive vertical dewarping to flatten paper curvature near the gutter."""
    h, w = image.shape[:2]
    if h < 260 or w < 260:
        return image

    gray = preprocess_image(image)
    col_mean = gray.mean(axis=0)
    center = w // 2
    search = max(20, int(w * 0.2))
    s0, s1 = max(0, center - search), min(w, center + search)
    if s1 - s0 < 12:
        return image

    local = col_mean[s0:s1]
    gutter_x = s0 + int(np.argmin(local))

    # Stronger at sides, weaker at gutter. Strength adapts with image size.
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    x_norm = np.abs((x_coords - np.float32(gutter_x)) / np.float32(max(1.0, w / 2.0)))
    base_strength = float(np.clip(0.022 + (h / 3000.0), 0.02, 0.04))
    y_offset = (x_norm**2) * np.float32(base_strength * h)

    map_x = np.ascontiguousarray(x_coords, dtype=np.float32)
    map_y = np.ascontiguousarray(np.clip(y_coords - y_offset, 0, h - 1), dtype=np.float32)

    # Build CV_32FC2 map explicitly; this avoids platform-specific map1/map2 typing issues.
    map_xy = np.dstack((map_x, map_y)).astype(np.float32, copy=False)
    map_xy = np.ascontiguousarray(map_xy)

    try:
        return cv2.remap(
            image,
            map_xy,
            None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        # Safe fallback: keep pipeline running even if a local OpenCV build is picky.
        return image


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
    filename = output_dir / f"page_{index:04d}.{OUTPUT_FORMAT}"
    cv2.imwrite(str(filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    logging.info("Saved %s", filename)

def save_pages_as_pdf(images: List[np.ndarray], output_dir: Path, filename: str) -> None:
    """Save all processed pages in order as one PDF file."""
    if not images:
        logging.warning("No processed pages available for PDF export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / filename

    pil_pages = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
    first_page = pil_pages[0]
    first_page.save(pdf_path, save_all=True, append_images=pil_pages[1:])
    logging.info("Saved %s", pdf_path)

def process_page(image: np.ndarray) -> np.ndarray:
    """Process a single page through crop, perspective correction, dewarp, and lighting."""
    aligned = align_book_image(image)
    corrected = correct_perspective(aligned) if ENABLE_PERSPECTIVE_CORRECTION else aligned
    cropped = crop_page(corrected) if ENABLE_CROP else corrected
    dewarped = dewarp_page(cropped) if ENABLE_DEWARP else cropped
    normalized = (
        normalize_lighting(dewarped) if ENABLE_LIGHTING_NORMALIZATION else dewarped
    )
    return normalized


def main() -> None:
    """Entry point for batch processing."""
    image_paths = load_images(INPUT_DIR)
    if not image_paths:
        logging.error("No images to process.")
        return

    page_index = 1
    processed_pages: List[np.ndarray] = []
    for image_path in image_paths:
        logging.info("Processing %s", image_path)
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.error("Failed to read image: %s", image_path)
                continue

            image = align_book_image(image)

            pages = [image]
            if is_double_page(image):
                pages = list(split_double_page(image))

            for page in pages:
                processed = process_page(page)
                if processed.size == 0:
                    logging.warning("Skipping empty result for %s", image_path)
                    continue
                if OUTPUT_AS_PDF:
                    processed_pages.append(processed)
                else:
                    save_page(processed, OUTPUT_DIR, page_index)
                page_index += 1

        except Exception as exc:
            logging.exception("Error processing %s: %s", image_path, exc)

    if OUTPUT_AS_PDF:
        save_pages_as_pdf(processed_pages, OUTPUT_DIR, PDF_FILENAME)


if __name__ == "__main__":
    main()
