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
PDF_FILENAME = "book_scan2.pdf"
JPEG_QUALITY = 60  # normaler/empfohlener Wert: 80-90 | verlustreich (lossy)
PDF_IMAGE_QUALITY = 75  # normaler/empfohlener Wert: 70-85 | verlustreich (lossy)
PDF_RESOLUTION_DPI = 200  # normaler/empfohlener Wert: 150-250 | verlustreich bei Reduktion
PDF_SOURCE_DPI = 300  # normaler/empfohlener Wert: 300 | Referenz-DPI für PDF-Downscaling

ENABLE_PERSPECTIVE_CORRECTION = True
ENABLE_CROP = True
ENABLE_DEWARP = True
DEWARP_THRESHOLD = 50

# Optional YOLO-based page detection (recommended for unstable contour detections).
ENABLE_YOLO_PAGE_DETECTION = True
YOLO_MODEL_PATH = "yolo26s.pt"  # Can also be a custom page detector model.
YOLO_CONFIDENCE = 0.75
YOLO_IOU = 0.70
YOLO_TARGET_CLASSES: Optional[List[str]] = ["book"]
YOLO_MIN_AREA_RATIO = 0.30
YOLO_MIN_SIDE_RATIO = 0.35
YOLO_MIN_MASK_COVERAGE = 0.60
# Weight for preferring detections that cover the image center column.
# This is more robust than relying only on the YOLO box midpoint.
YOLO_CENTER_WEIGHT = 0.70
YOLO_MIN_RELATIVE_TO_CONTOUR = 0.75
YOLO_MASTER_MODE = True
YOLO_MASTER_MIN_IOU_FOR_CONTOUR_EXPAND = 0.10
YOLO_MASTER_MAX_CONTOUR_EXPAND = 0.13

# Split-line tuning: keep page separation centered in the whole image.
SPLIT_SEARCH_WINDOW_RATIO = 0.24
SPLIT_IMAGE_CENTER_WEIGHT = 0.90
SPLIT_CENTER_BIAS_WEIGHT = 0.80


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


_YOLO_MODEL = None
_YOLO_LOAD_ATTEMPTED = False


def _get_yolo_model():
    """Lazy-load an Ultralytics YOLO model if enabled and available."""
    global _YOLO_MODEL, _YOLO_LOAD_ATTEMPTED
    if not ENABLE_YOLO_PAGE_DETECTION:
        return None
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if _YOLO_LOAD_ATTEMPTED:
        return None

    _YOLO_LOAD_ATTEMPTED = True
    try:
        from ultralytics import YOLO  # type: ignore

        _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        logging.info("Loaded YOLO model: %s", YOLO_MODEL_PATH)
    except Exception as exc:
        logging.warning("YOLO page detection disabled (load failed): %s", exc)
        _YOLO_MODEL = None
    return _YOLO_MODEL


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union between two x0,y0,x1,y1 boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, (ax1 - ax0) * (ay1 - ay0)))
    area_b = float(max(1, (bx1 - bx0) * (by1 - by0)))
    return inter / max(1e-6, area_a + area_b - inter)


def _mask_coverage(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """How much of the bbox is supported by the page mask."""
    x0, y0, x1, y1 = bbox
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


def _center_column_bias(bbox: Tuple[int, int, int, int], image_w: int) -> float:
    """Score how well bbox aligns with the global image center column (0..1)."""
    x0, _, x1, _ = bbox
    image_center_x = image_w / 2.0

    # Strong prior: best if the actual image center lies inside the detection box.
    center_inside = 1.0 if x0 <= image_center_x <= x1 else 0.0

    # Smooth fallback: if not inside, prefer boxes close to image center.
    if center_inside > 0.0:
        distance_score = 1.0
    else:
        edge_distance = min(abs(image_center_x - x0), abs(image_center_x - x1))
        distance_score = max(0.0, 1.0 - (edge_distance / max(1.0, image_w / 2.0)))

    # "center_inside" dominates; distance_score helps when no candidate covers center.
    return 0.85 * center_inside + 0.15 * distance_score




def _largest_mask_component_bbox(mask: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Find mask-component bbox with strongest overlap to roi (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0
    roi_area = float(max(1, (x1 - x0) * (y1 - y0)))

    for i in range(1, num_labels):
        lx, ly, lw, lh, area = stats[i]
        if area <= 0:
            continue
        bx0, by0 = int(lx), int(ly)
        bx1, by1 = int(lx + lw), int(ly + lh)
        inter = _bbox_iou((x0, y0, x1, y1), (bx0, by0, bx1, by1))

        # Prefer components that overlap strongly and are not tiny versus YOLO ROI.
        rel_area = float(area) / roi_area
        score = 0.75 * inter + 0.25 * min(rel_area, 1.6)
        if score > best_score:
            best_score = score
            best_bbox = (bx0, by0, bx1, by1)

    return best_bbox



def _expand_bbox_towards(
    master_bbox: Tuple[int, int, int, int],
    helper_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    max_expand_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand master bbox slightly towards helper bbox, bounded by max_expand_ratio."""
    h, w = image_shape
    mx0, my0, mx1, my1 = master_bbox
    hx0, hy0, hx1, hy1 = helper_bbox

    mw = max(1, mx1 - mx0)
    mh = max(1, my1 - my0)
    lim_x = int(mw * max_expand_ratio)
    lim_y = int(mh * max_expand_ratio)

    nx0 = max(0, mx0 - min(lim_x, max(0, mx0 - hx0)))
    ny0 = max(0, my0 - min(lim_y, max(0, my0 - hy0)))
    nx1 = min(w, mx1 + min(lim_x, max(0, hx1 - mx1)))
    ny1 = min(h, my1 + min(lim_y, max(0, hy1 - my1)))
    return nx0, ny0, nx1, ny1


def _detect_book_region_yolo(
    image: np.ndarray,
    mask: np.ndarray,
    expected_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Detect primary page/book box with YOLO, filtered by geometry and mask agreement."""
    model = _get_yolo_model()
    if model is None:
        return None

    try:
        result = model.predict(image, conf=YOLO_CONFIDENCE, iou=YOLO_IOU, verbose=False)[0]
    except Exception as exc:
        logging.warning("YOLO inference failed, using contour fallback: %s", exc)
        return None

    if result.boxes is None or len(result.boxes) == 0:
        return None

    names = getattr(result, "names", {}) or {}
    h, w = image.shape[:2]
    allowed = {name.lower() for name in YOLO_TARGET_CLASSES} if YOLO_TARGET_CLASSES else None

    best_box: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0
    for box in result.boxes:
        cls_idx = int(box.cls[0].item()) if box.cls is not None else -1
        cls_name = str(names.get(cls_idx, "")).lower()
        if allowed and cls_name and cls_name not in allowed:
            continue

        x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().tolist()
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue

        area_ratio = (bw * bh) / float(max(1, h * w))
        if area_ratio < YOLO_MIN_AREA_RATIO:
            continue
        if bw < int(w * YOLO_MIN_SIDE_RATIO) or bh < int(h * YOLO_MIN_SIDE_RATIO):
            continue

        aspect = bw / float(max(1, bh))
        if aspect > 2.25 or aspect < 0.38:
            continue

        candidate = (x0, y0, x1, y1)
        coverage = _mask_coverage(mask, candidate)
        if coverage < YOLO_MIN_MASK_COVERAGE:
            continue

        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        center_bias = _center_column_bias(candidate, w)
        score = (
            (conf * 0.50)
            + (area_ratio * 0.20)
            + (coverage * 0.30)
            + (YOLO_CENTER_WEIGHT * center_bias)
        )

        if expected_bbox is not None:
            score += 0.35 * _bbox_iou(candidate, expected_bbox)

        if score > best_score:
            best_score = score
            best_box = candidate

    if best_box is None:
        return None

    x0, y0, x1, y1 = best_box
    bw, bh = x1 - x0, y1 - y0
    pad_x = max(12, int(bw * 0.02))
    pad_y_top = max(10, int(bh * 0.018))
    pad_y_bottom = max(18, int(bh * 0.05))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y_top)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y_bottom)

    contour = np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )
    return contour, (x0, y0, x1, y1)


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


def _filter_mask_components(mask: np.ndarray) -> np.ndarray:
    """Drop tiny/slim components that are likely illustrations or background strips."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    out = np.zeros_like(mask)
    min_area = h * w * 0.012
    min_h = h * 0.16
    min_w = w * 0.10

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_w or bh < min_h:
            continue
        aspect = bw / float(max(1, bh))
        if aspect > 3.2 or aspect < 0.18:
            continue
        cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)

    return out if np.any(out) else mask


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

    return _filter_mask_components(mask)


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

    moments = cv2.moments(contour)
    if abs(moments['m00']) < 1e-6:
        cx, cy = x + (bw / 2.0), y + (bh / 2.0)
    else:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

    center_dx = abs(cx - (image_w / 2.0)) / max(1.0, image_w / 2.0)
    center_dy = abs(cy - (image_h / 2.0)) / max(1.0, image_h / 2.0)
    center_score = max(0.0, 1.0 - (0.7 * center_dx + 0.3 * center_dy))

    # Page-like contours should have significant area, be reasonably rectangular,
    # and usually extend low in the image (useful against chapter-heading/text blobs).
    aspect_score = max(0.0, 1.0 - min(abs(aspect - 0.75), abs(aspect - 1.35)) / 1.35)
    bottom_reach = (y + bh) / float(max(1, image_h))

    # Penalize stripe-like detections.
    strip_penalty = 0.0
    if aspect > 2.6 or aspect < 0.28:
        strip_penalty = 0.6

    return (
        2.9 * area_ratio
        + 1.0 * fill_ratio
        + 0.55 * aspect_score
        + 0.35 * bottom_reach
        + 0.75 * center_score
        - strip_penalty
    )


def _select_best_page_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Select the most page-like contour from the cleaned mask."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = h * w * 0.10
    candidates = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < w * 0.22 or bh < h * 0.30:
            continue
        candidates.append(c)

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


def detect_book_region_from_image(image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Detect main page/book region with YOLO as master and contour as optional helper."""
    mask = build_page_mask(image)
    contour_region = detect_book_region(mask)

    if not ENABLE_YOLO_PAGE_DETECTION:
        return contour_region

    expected_bbox = contour_region[1] if contour_region is not None else None
    yolo_region = _detect_book_region_yolo(image, mask, expected_bbox=expected_bbox)
    if yolo_region is None:
        return contour_region

    _, yolo_bbox = yolo_region

    # Snap YOLO to the strongest overlapping mask component; avoids illustration crops.
    component_bbox = _largest_mask_component_bbox(mask, yolo_bbox)
    if component_bbox is not None:
        cx0, cy0, cx1, cy1 = component_bbox
        yolo_bbox = (cx0, cy0, cx1, cy1)

    if YOLO_MASTER_MODE:
        final_bbox = yolo_bbox
        if contour_region is not None:
            _, contour_bbox = contour_region
            overlap = _bbox_iou(yolo_bbox, contour_bbox)
            if overlap >= YOLO_MASTER_MIN_IOU_FOR_CONTOUR_EXPAND:
                final_bbox = _expand_bbox_towards(
                    yolo_bbox,
                    contour_bbox,
                    image.shape[:2],
                    YOLO_MASTER_MAX_CONTOUR_EXPAND,
                )
        fx0, fy0, fx1, fy1 = final_bbox
        contour = np.array([[[fx0, fy0]], [[fx1, fy0]], [[fx1, fy1]], [[fx0, fy1]]], dtype=np.int32)
        return contour, final_bbox

    if contour_region is None:
        cx0, cy0, cx1, cy1 = yolo_bbox
        contour = np.array([[[cx0, cy0]], [[cx1, cy0]], [[cx1, cy1]], [[cx0, cy1]]], dtype=np.int32)
        return contour, yolo_bbox

    _, contour_bbox = contour_region
    yolo_area = float(max(1, (yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1])))
    contour_area = float(max(1, (contour_bbox[2] - contour_bbox[0]) * (contour_bbox[3] - contour_bbox[1])))
    area_ratio = yolo_area / contour_area

    overlap = _bbox_iou(yolo_bbox, contour_bbox)
    yolo_cov = _mask_coverage(mask, yolo_bbox)

    if area_ratio < YOLO_MIN_RELATIVE_TO_CONTOUR:
        return contour_region

    if overlap < 0.28 and yolo_cov < 0.62:
        return contour_region

    cx0, cy0, cx1, cy1 = yolo_bbox
    contour = np.array([[[cx0, cy0]], [[cx1, cy0]], [[cx1, cy1]], [[cx0, cy1]]], dtype=np.int32)
    return contour, yolo_bbox


def detect_page_boxes(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect candidate page boxes sorted left-to-right."""
    mask = build_page_mask(image)
    h, w = mask.shape[:2]

    region = detect_book_region_from_image(image)
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
    """Estimate center gutter x-position with a strong full-image center prior."""
    gray = preprocess_image(image)
    _, w = gray.shape[:2]

    image_center = w / 2.0
    center = image_center
    region = detect_book_region_from_image(image)
    if region is not None:
        _, (bx0, _, bx1, _) = region
        book_center = (bx0 + bx1) / 2.0
        # Keep split estimation anchored to the whole image center.
        center = (SPLIT_IMAGE_CENTER_WEIGHT * image_center) + ((1.0 - SPLIT_IMAGE_CENTER_WEIGHT) * book_center)

    window = max(20, int(w * SPLIT_SEARCH_WINDOW_RATIO))
    center_i = int(round(center))
    start = max(0, center_i - window)
    end = min(w, center_i + window)

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
    # Strong center prior: keep split line near the middle of the full image.
    score = darkness + 0.85 * edge_strength + SPLIT_CENTER_BIAS_WEIGHT * center_bias * np.max(darkness + 1e-6)

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
    if has_center_seam:
        return True

    # Expensive fallback only when seam signal is weak/ambiguous.
    boxes = detect_page_boxes(image)
    return len(boxes) == 2


def split_double_page(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a double-page image into left and right pages with overlap safety."""
    h, w = image.shape[:2]
    seam_x, _ = find_split_line(image)
    region = detect_book_region_from_image(image)
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
    if ENABLE_YOLO_PAGE_DETECTION:
        region = detect_book_region_from_image(image)
        if region is not None:
            _, (x0, y0, x1, y1) = region
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

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
    region = detect_book_region_from_image(image)
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
    """Dewarp page by fitting a stable rotated page rectangle and warping to top view."""
    h, w = image.shape[:2]
    if h < 220 or w < 220:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, DEWARP_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    page_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(page_contour)
    if area < (h * w * 0.30):
        return image

    rect = cv2.minAreaRect(page_contour)
    box = cv2.boxPoints(rect).astype(np.float32)

    sums = box.sum(axis=1)
    tl = box[np.argmin(sums)]
    br = box[np.argmax(sums)]

    diffs = np.diff(box, axis=1)
    tr = box[np.argmin(diffs)]
    bl = box[np.argmax(diffs)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 200 or max_height < 200:
        return image

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)


def save_page(image: np.ndarray, output_dir: Path, index: int) -> None:
    """Save the processed page image with sequential naming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"page_{index:04d}.jpg"

    try:
        ok = cv2.imwrite(
            str(filename),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(JPEG_QUALITY, 0, 100))],
        )
        if not ok:
            raise OSError("cv2.imwrite returned False")

    except Exception as exc:
        logging.error("Failed to save %s: %s", filename, exc)
        return

    logging.info("Saved %s", filename)


def _resize_for_pdf_dpi(image: np.ndarray) -> np.ndarray:
    """Resize page image to requested PDF dpi using PDF_SOURCE_DPI as reference."""
    target_dpi = float(max(72, PDF_RESOLUTION_DPI))
    source_dpi = float(max(72, PDF_SOURCE_DPI))

    scale = target_dpi / source_dpi
    if scale >= 1.0:
        return image

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_pages_as_pdf(images: List[np.ndarray], output_dir: Path, filename: str) -> None:
    """Save all processed pages in order as one PDF file."""
    if not images:
        logging.warning("No processed pages available for PDF export.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / filename

    prepared_images = [_resize_for_pdf_dpi(image) for image in images]
    pil_pages = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in prepared_images]
    first_page = pil_pages[0]
    first_page.save(
        pdf_path,
        save_all=True,
        append_images=pil_pages[1:],
        resolution=float(max(72, PDF_RESOLUTION_DPI)),
        quality=int(np.clip(PDF_IMAGE_QUALITY, 1, 95)),
        optimize=True,
    )
    logging.info("Saved %s", pdf_path)


def process_page(image: np.ndarray) -> np.ndarray:
    """Process a single page through perspective correction, crop, and dewarp."""
    corrected = correct_perspective(image) if ENABLE_PERSPECTIVE_CORRECTION else image
    cropped = crop_page(corrected) if ENABLE_CROP else corrected
    dewarped = dewarp_page(cropped) if ENABLE_DEWARP else cropped
    return dewarped


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

            # Keep split detection on original image for speed before page processing.
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