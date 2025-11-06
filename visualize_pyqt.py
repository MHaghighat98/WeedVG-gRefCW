"""
A professional, high-quality annotation visualizer for the CropAndWeed dataset,
built with PyQt6 for a robust and responsive user experience.
"""

import sys
import json
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QFrame,
    QComboBox,
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt

# --- Configuration ---
# Try multiple likely locations so the visualizer works whether annotations
# live under the 'Data/grefcoco_format' folder or directly in this workspace.
_CANDIDATE_GREFS = [
    r"c:\\Users\\jd138001\\Downloads\\Combination\\grefs(unc).json",
    r"c:\\Users\\jd138001\\Downloads\\Data\\grefcoco_format\\grefs(unc).json",
]
_CANDIDATE_INSTANCES = [
    r"c:\\Users\\jd138001\\Downloads\\Combination\\instances.json",
    r"c:\\Users\\jd138001\\Downloads\\Data\\grefcoco_format\\instances.json",
]
_CANDIDATE_IMAGES = [
    r"c:\\Users\\jd138001\\Downloads\\data\\images",
    r"c:\\Users\\jd138001\\Downloads\\Combination\\data\\images",
]


def _resolve_first(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return paths[0]


GREFS_JSON = _resolve_first(_CANDIDATE_GREFS)
INSTANCES_JSON = _resolve_first(_CANDIDATE_INSTANCES)
IMAGES_DIR = _resolve_first(_CANDIDATE_IMAGES)
CATEGORY_NAMES = {1: "crop", 2: "weed"}
CATEGORY_COLORS = {
    "crop": QColor(40, 167, 69),  # A nice, modern green
    "weed": QColor(220, 53, 69),  # A strong but not jarring red
    "unknown": QColor(108, 117, 125),  # Gray
}


# --- Data Loading ---
def load_annotations():
    """Load and merge gRefCOCO and COCO instance annotations.

    This function is defensive about file locations and sentence key variants
    (e.g. 'sentence', 'raw', 'sent', 'text'). For the test split it guarantees
    that each instance will have a 'test sentence' key and an 'original_sentence'
    and change metadata so the UI panels can always display modifications.
    """
    print(f"Loading grefs from {GREFS_JSON}...")
    with open(GREFS_JSON, "r", encoding="utf-8") as f:
        grefs_data = json.load(f)
    print(f"✅ Loaded {len(grefs_data.get('images', []))} images")

    print(f"Loading instances from {INSTANCES_JSON}...")
    with open(INSTANCES_JSON, "r", encoding="utf-8") as f:
        instances_data = json.load(f)
    print(f"✅ Loaded {len(instances_data.get('annotations', []))} annotations")

    ann_id_to_bbox = {ann["id"]: ann for ann in instances_data.get("annotations", [])}

    # Helper to get the best sentence string for an instance
    def _get_sentence(inst):
        for k in ("sentence", "raw", "sent", "text"):
            v = inst.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            # sometimes sentences are dicts like {"raw": "..."}
            if isinstance(v, dict) and v.get("raw"):
                return str(v.get("raw")).strip()
        return None

    for img in grefs_data.get("images", []):
        # Attach bbox, category_name and crop_name to instances when possible
        for inst in img.get("instance_sentences", []):
            ann_id = inst.get("ann_id")
            if ann_id in ann_id_to_bbox:
                ann = ann_id_to_bbox[ann_id]
                inst["bbox"] = ann.get("bbox", inst.get("bbox"))
                inst["category_name"] = CATEGORY_NAMES.get(inst.get("category_id"), inst.get("category_name", "unknown"))
                inst["crop_name"] = ann.get("crop_name", inst.get("crop_name", "unknown"))

            # Normalize/ensure sentence fields exist for later processing
            sentence = _get_sentence(inst)
            if sentence:
                # keep original 'sentence' key for backward compatibility
                inst.setdefault("sentence", sentence)

    return grefs_data.get("images", [])


# --- PyQt6 Components ---


class ImageDisplay(QLabel):
    """A custom QLabel to display the image and draw bounding boxes."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333; border-radius: 5px;")
        self.image_data = None
        self.pixmap = None
        # Interactive view state
        self.fit_to_window = False  # show original resolution by default
        self.scale_factor = 1.0  # absolute when not fitting; relative when fitting
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._panning = False
        self._last_pos = None

    def set_image(self, image_data):
        self.image_data = image_data
        img_path = os.path.join(IMAGES_DIR, image_data["file_name"])
        if os.path.exists(img_path):
            self.pixmap = QPixmap(img_path)
            if self.pixmap.isNull():
                self.setText(f"Failed to load image:\n{image_data['file_name']}")
                self.pixmap = None
                return
            # Reset view for new image, respect current mode
            self.scale_factor = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.update_display()
        else:
            self.setText(f"Image not found:\n{image_data['file_name']}")
            self.pixmap = None

    def update_display(self):
        # Trigger repaint using current state
        self.update()

    def _fit_scale(self):
        """Scale that fits the image to the widget while preserving aspect ratio."""
        if not self.pixmap:
            return 1.0
        pw, ph = self.pixmap.width(), self.pixmap.height()
        if pw == 0 or ph == 0 or self.width() == 0 or self.height() == 0:
            return 1.0
        return min(self.width() / pw, self.height() / ph)

    def _effective_scale(self):
        if self.fit_to_window:
            return self._fit_scale() * max(0.1, min(self.scale_factor, 20.0))
        return max(0.05, min(self.scale_factor, 20.0))

    def _image_origin(self, scale):
        """Top-left display origin (x0,y0) when image is centered, before pan."""
        pw, ph = self.pixmap.width(), self.pixmap.height()
        vw, vh = self.width(), self.height()
        img_w, img_h = pw * scale, ph * scale
        x0 = (vw - img_w) / 2.0
        y0 = (vh - img_h) / 2.0
        return x0, y0

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap or self.width() == 0 or self.height() == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        scale = self._effective_scale()
        base_x0, base_y0 = self._image_origin(scale)
        x0 = base_x0 + self.pan_x
        y0 = base_y0 + self.pan_y

        # Draw image
        pw, ph = self.pixmap.width(), self.pixmap.height()
        painter.drawPixmap(int(x0), int(y0), int(pw * scale), int(ph * scale), self.pixmap)

        # Draw boxes
        for idx, inst in enumerate(self.image_data.get("instance_sentences", [])):
            if "bbox" not in inst:
                continue
            x, y, w, h = inst["bbox"]
            category = inst.get("category_name", "unknown")
            pen = QPen(CATEGORY_COLORS.get(category, CATEGORY_COLORS["unknown"]), 3)
            painter.setPen(pen)
            rx = x0 + x * scale
            ry = y0 + y * scale
            rw = w * scale
            rh = h * scale
            painter.drawRect(int(rx), int(ry), int(rw), int(rh))

            # Label background and text
            label_w, label_h = 24, 18
            painter.fillRect(int(rx), int(ry) - label_h, label_w, label_h, pen.color())
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(int(rx) + 6, int(ry) - 4, str(idx + 1))

    def resizeEvent(self, event):
        # Only update fit if we are in fit-to-window mode
        self.update_display()

    def set_fit_to_window(self, fit: bool):
        self.fit_to_window = fit
        # Reset zoom and pan when switching modes for predictable behavior
        self.scale_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def reset_to_original(self):
        # Disable fit and set absolute scale to 1.0 (1 image pixel == 1 screen pixel)
        self.fit_to_window = False
        self.scale_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pos = event.position()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pos is not None:
            cur = event.position()
            delta = cur - self._last_pos
            self.pan_x += float(delta.x())
            self.pan_y += float(delta.y())
            self._last_pos = cur
            self.update()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._last_pos = None
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event):
        """Zoom with mouse wheel around the cursor position."""
        if not self.pixmap:
            return
        cursor_pos = event.position()
        cx, cy = float(cursor_pos.x()), float(cursor_pos.y())

        # Current
        old_scale = self._effective_scale()
        base_x0, base_y0 = self._image_origin(old_scale)
        x0 = base_x0 + self.pan_x
        y0 = base_y0 + self.pan_y

        # Image coords under cursor
        u = (cx - x0) / old_scale
        v = (cy - y0) / old_scale

        # Update zoom factor
        angle = event.angleDelta().y()
        if angle > 0:
            self.scale_factor *= 1.15
        else:
            self.scale_factor /= 1.15
        self.scale_factor = max(0.1, min(self.scale_factor, 20.0))

        # New
        new_scale = self._effective_scale()
        new_base_x0, new_base_y0 = self._image_origin(new_scale)

        # Adjust pan so the same image point stays under cursor
        self.pan_x = (cx - u * new_scale) - new_base_x0
        self.pan_y = (cy - v * new_scale) - new_base_y0

        self.update()


class AnnotationPanel(QScrollArea):
    """A scrollable panel to display all annotations and metadata."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #f8f9fa; border-radius: 5px;")

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.container)

    def update_content(self, image_data):
        # Clear old content
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        # --- Image Info ---
        title = QLabel("Image Information")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #343a40; margin-bottom: 10px;")
        self.layout.addWidget(title)

        info_text = (
            f"<b>File:</b> {image_data['file_name']}<br><b>ID:</b> {image_data['id']} | <b>Size:</b> {image_data['width']}x{image_data['height']}px"
        )
        self.layout.addWidget(QLabel(info_text))

        # --- Category Counts ---
        instances = image_data.get("instance_sentences", [])
        crop_count = sum(1 for inst in instances if inst.get("category_name") == "crop")
        weed_count = sum(1 for inst in instances if inst.get("category_name") == "weed")

        counts_label = QLabel(f"<b>Crops:</b> {crop_count} | <b>Weeds:</b> {weed_count}")
        counts_label.setStyleSheet("margin-top: 10px; font-size: 14px;")
        self.layout.addWidget(counts_label)

        # --- Image-level Sentences (robust across formats) ---
        img_level_sentences = []
        # Common patterns seen in datasets
        if isinstance(image_data.get("sentences"), list):
            for s in image_data["sentences"]:
                if isinstance(s, dict) and "raw" in s:
                    img_level_sentences.append(str(s["raw"]))
                else:
                    img_level_sentences.append(str(s))
        elif isinstance(image_data.get("image_level_sentences"), list):
            for s in image_data["image_level_sentences"]:
                if isinstance(s, dict) and "raw" in s:
                    img_level_sentences.append(str(s["raw"]))
                else:
                    img_level_sentences.append(str(s))

        # Negative sentence (what's absent) if present
        if isinstance(image_data.get("negative_sentence"), str) and image_data["negative_sentence"].strip():
            img_level_sentences.append(f"[Negative] {image_data['negative_sentence']}")

        if img_level_sentences:
            img_level_title = QLabel("Image-Level Sentences")
            img_level_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #343a40; margin-top: 20px; margin-bottom: 10px;")
            self.layout.addWidget(img_level_title)

            for s in img_level_sentences:
                sent_label = QLabel(f'<i>"{s}"</i>')
                sent_label.setWordWrap(True)
                self.layout.addWidget(sent_label)

        # --- Crop Distribution Summary ---
        # Count crop types in this image
        from collections import Counter

        crop_counts = Counter()
        for inst in instances:
            if inst.get("category_name") == "crop":
                crop_name = inst.get("crop_name", "unknown")
                crop_counts[crop_name] += 1

        weed_count = sum(1 for inst in instances if inst.get("category_name") == "weed")

        if crop_counts or weed_count > 0:
            summary_title = QLabel("Crop Distribution")
            summary_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2; margin-top: 20px; margin-bottom: 10px;")
            self.layout.addWidget(summary_title)

            # Create summary text
            summary_parts = []
            for crop_name, count in sorted(crop_counts.items()):
                summary_parts.append(f"<b>{crop_name}:</b> {count}")
            if weed_count > 0:
                summary_parts.append(f"<b>Weeds:</b> {weed_count}")

            summary_text = " | ".join(summary_parts)
            summary_label = QLabel(summary_text)
            summary_label.setWordWrap(True)
            summary_label.setStyleSheet("padding: 10px; background-color: #E3F2FD; border-radius: 4px; color: #0D47A1;")
            self.layout.addWidget(summary_label)

        # --- Instances ---
        instances_title = QLabel(f"Instances ({len(instances)})")
        instances_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #343a40; margin-top: 20px; margin-bottom: 10px;")
        self.layout.addWidget(instances_title)

        for idx, inst in enumerate(instances):
            category = inst.get("category_name", "unknown")
            crop_name = inst.get("crop_name", "unknown")
            color = CATEGORY_COLORS[category].name()

            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setStyleSheet(f"background-color: white; border-left: 4px solid {color}; border-radius: 4px; margin-bottom: 10px;")

            vbox = QVBoxLayout(frame)

            # Create header with crop type for crops, category for weeds
            if category == "crop":
                header = f"<b>[{idx + 1}] {crop_name.upper()}</b>"
            else:
                header = f"<b>[{idx + 1}] {category.upper()}</b>"
            vbox.addWidget(QLabel(header))

            # Show annotation ID
            ann_id = inst.get("ann_id", "N/A")
            ann_id_label = QLabel(f"<small>Ann ID: {ann_id}</small>")
            ann_id_label.setStyleSheet("color: #6c757d;")
            vbox.addWidget(ann_id_label)

            bbox = inst.get("bbox", ["N/A"] * 4)
            bbox_text = f"BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            vbox.addWidget(QLabel(bbox_text))

            # Show sentence for this instance
            # Check if this is a test split image to show test sentence information
            is_test_split = image_data.get("split", "").lower() == "test"

            if is_test_split:
                # For test split, show test sentence, original sentence, and changes
                original_sentence = inst.get("original_sentence")
                test_sentence = inst.get("test_sentence")

                if test_sentence:
                    test_label = QLabel(f'<b>Test Sentence:</b> <i>"{test_sentence}"</i>')
                    test_label.setWordWrap(True)
                    test_label.setStyleSheet("color: #cc0000; margin-top: 5px; font-weight: bold;")
                    vbox.addWidget(test_label)

                    # Show original sentence if it exists and is different
                    if original_sentence and original_sentence != test_sentence:
                        orig_label = QLabel(f'<b>Original Sentence:</b> <i>"{original_sentence}"</i>')
                        orig_label.setWordWrap(True)
                        orig_label.setStyleSheet("color: #006600; margin-top: 5px; font-weight: bold;")
                        vbox.addWidget(orig_label)

                    # Show change metadata if available
                    if inst.get("change_type"):
                        change_detail = inst.get("change_detail", {})
                        change_text = f"<small><b>Change:</b> {inst['change_type']}"
                        if change_detail:
                            change_text += f" ({change_detail.get('attribute', 'N/A')}: "
                            change_text += f"{change_detail.get('from', 'N/A')} → {change_detail.get('to', 'N/A')})"
                        change_text += "</small>"
                        change_label = QLabel(change_text)
                        change_label.setStyleSheet("color: #856404; margin-top: 3px;")
                        vbox.addWidget(change_label)
                else:
                    # Fallback to regular sentence for test items without modifications
                    sentence = inst.get("sentence") or inst.get("original_sentence") or inst.get("raw") or inst.get("sent") or inst.get("text")
                    if sentence:
                        sent_label = QLabel(f'<b>Sentence:</b> <i>"{sentence}"</i>')
                        sent_label.setWordWrap(True)
                        sent_label.setStyleSheet("color: #006600; margin-top: 5px;")
                        vbox.addWidget(sent_label)
                    else:
                        vbox.addWidget(QLabel("<i>No sentence available</i>"))
            else:
                # Regular handling for train/val splits
                sentence = inst.get("sentence") or inst.get("original_sentence") or inst.get("raw") or inst.get("sent") or inst.get("text")
                if sentence:
                    sent_label = QLabel(f'<b>Sentence:</b> <i>"{sentence}"</i>')
                    sent_label.setWordWrap(True)
                    sent_label.setStyleSheet("color: #006600; margin-top: 5px;")
                    vbox.addWidget(sent_label)
                else:
                    vbox.addWidget(QLabel("<i>No sentence available</i>"))

            self.layout.addWidget(frame)


class TestAnnotationPanel(QScrollArea):
    """A scrollable panel for test split that shows both sentence and original_sentence."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background-color: #fff3cd; border-radius: 5px;")  # Light yellow for test

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.container)

    def update_content(self, image_data):
        # Clear old content
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        # --- Image Info ---
        title = QLabel("Image Information (TEST SPLIT)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #856404; margin-bottom: 10px;")
        self.layout.addWidget(title)

        info_text = (
            f"<b>File:</b> {image_data['file_name']}<br><b>ID:</b> {image_data['id']} | <b>Size:</b> {image_data['width']}x{image_data['height']}px"
        )
        self.layout.addWidget(QLabel(info_text))

        # --- Category Counts ---
        instances = image_data.get("instance_sentences", [])
        crop_count = sum(1 for inst in instances if inst.get("category_name") == "crop")
        weed_count = sum(1 for inst in instances if inst.get("category_name") == "weed")

        counts_label = QLabel(f"<b>Crops:</b> {crop_count} | <b>Weeds:</b> {weed_count}")
        counts_label.setStyleSheet("margin-top: 10px; font-size: 14px;")
        self.layout.addWidget(counts_label)

        # --- Image-level Sentences ---
        img_level_sentences = []
        if isinstance(image_data.get("sentences"), list):
            for s in image_data["sentences"]:
                if isinstance(s, dict) and "raw" in s:
                    img_level_sentences.append(str(s["raw"]))
                else:
                    img_level_sentences.append(str(s))
        elif isinstance(image_data.get("image_level_sentences"), list):
            for s in image_data["image_level_sentences"]:
                if isinstance(s, dict) and "raw" in s:
                    img_level_sentences.append(str(s["raw"]))
                else:
                    img_level_sentences.append(str(s))

        if isinstance(image_data.get("negative_sentence"), str) and image_data["negative_sentence"].strip():
            img_level_sentences.append(f"[Negative] {image_data['negative_sentence']}")

        if img_level_sentences:
            img_level_title = QLabel("Image-Level Sentences")
            img_level_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #856404; margin-top: 20px; margin-bottom: 10px;")
            self.layout.addWidget(img_level_title)

            for s in img_level_sentences:
                sent_label = QLabel(f'<i>"{s}"</i>')
                sent_label.setWordWrap(True)
                self.layout.addWidget(sent_label)

        # --- Crop Distribution Summary ---
        # Count crop types in this image
        from collections import Counter

        crop_counts = Counter()
        for inst in instances:
            if inst.get("category_name") == "crop":
                crop_name = inst.get("crop_name", "unknown")
                crop_counts[crop_name] += 1

        weed_count_detail = sum(1 for inst in instances if inst.get("category_name") == "weed")

        if crop_counts or weed_count_detail > 0:
            summary_title = QLabel("Crop Distribution")
            summary_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #856404; margin-top: 20px; margin-bottom: 10px;")
            self.layout.addWidget(summary_title)

            # Create summary text
            summary_parts = []
            for crop_name, count in sorted(crop_counts.items()):
                summary_parts.append(f"<b>{crop_name}:</b> {count}")
            if weed_count_detail > 0:
                summary_parts.append(f"<b>Weeds:</b> {weed_count_detail}")

            summary_text = " | ".join(summary_parts)
            summary_label = QLabel(summary_text)
            summary_label.setWordWrap(True)
            summary_label.setStyleSheet("padding: 10px; background-color: #FFF3CD; border-radius: 4px; color: #856404;")
            self.layout.addWidget(summary_label)

        # --- Instances ---
        instances_title = QLabel(f"Instances ({len(instances)})")
        instances_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #856404; margin-top: 20px; margin-bottom: 10px;")
        self.layout.addWidget(instances_title)

        for idx, inst in enumerate(instances):
            category = inst.get("category_name", "unknown")
            crop_name = inst.get("crop_name", "unknown")
            color = CATEGORY_COLORS[category].name()

            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame.setStyleSheet(f"background-color: white; border-left: 4px solid {color}; border-radius: 4px; margin-bottom: 10px;")

            vbox = QVBoxLayout(frame)

            # Create header with crop type for crops, category for weeds
            if category == "crop":
                header = f"<b>[{idx + 1}] {crop_name.upper()}</b>"
            else:
                header = f"<b>[{idx + 1}] {category.upper()}</b>"
            vbox.addWidget(QLabel(header))

            # Show annotation ID
            ann_id = inst.get("ann_id", "N/A")
            ann_id_label = QLabel(f"<small>Ann ID: {ann_id}</small>")
            ann_id_label.setStyleSheet("color: #856404;")
            vbox.addWidget(ann_id_label)

            bbox = inst.get("bbox", ["N/A"] * 4)
            bbox_text = f"BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            vbox.addWidget(QLabel(bbox_text))

            # Show BOTH original_sentence and test sentence for test split
            original_sentence = inst.get("original_sentence")
            test_sentence = inst.get("test_sentence")  # Note: underscore in key name
            regular_sentence = inst.get("sentence")  # For non-test items

            # For test split items, always show the test sentence prominently
            if test_sentence:
                test_label = QLabel(f'<b>Test Sentence:</b> <i>"{test_sentence}"</i>')
                test_label.setWordWrap(True)
                test_label.setStyleSheet("color: #cc0000; margin-top: 5px; font-weight: bold;")
                vbox.addWidget(test_label)

                # Show original sentence if it exists and is different
                if original_sentence and original_sentence != test_sentence:
                    orig_label = QLabel(f'<b>Original Sentence:</b> <i>"{original_sentence}"</i>')
                    orig_label.setWordWrap(True)
                    orig_label.setStyleSheet("color: #006600; margin-top: 5px; font-weight: bold;")
                    vbox.addWidget(orig_label)

                # Show change metadata if available
                if inst.get("change_type"):
                    change_detail = inst.get("change_detail", {})
                    change_text = f"<small><b>Change:</b> {inst['change_type']}"
                    if change_detail:
                        change_text += f" ({change_detail.get('attribute', 'N/A')}: "
                        change_text += f"{change_detail.get('from', 'N/A')} → {change_detail.get('to', 'N/A')})"
                    change_text += "</small>"
                    change_label = QLabel(change_text)
                    change_label.setStyleSheet("color: #856404; margin-top: 3px;")
                    vbox.addWidget(change_label)

            # Regular sentence (for train/val or test items without modifications)
            elif regular_sentence:
                sent_label = QLabel(f'<b>Sentence:</b> <i>"{regular_sentence}"</i>')
                sent_label.setWordWrap(True)
                sent_label.setStyleSheet("color: #006600; margin-top: 5px;")
                vbox.addWidget(sent_label)

            # Fallback to other keys
            else:
                for key in ("raw", "sent", "text"):
                    if key in inst and inst[key]:
                        fallback_label = QLabel(f'<b>Sentence:</b> <i>"{inst[key]}"</i>')
                        fallback_label.setWordWrap(True)
                        fallback_label.setStyleSheet("color: #006600; margin-top: 5px;")
                        vbox.addWidget(fallback_label)
                        break
                else:
                    vbox.addWidget(QLabel("<i>No sentence available</i>"))

            self.layout.addWidget(frame)


class TestSplitWindow(QMainWindow):
    """A separate window dedicated to viewing the test split."""

    def __init__(self, test_images):
        super().__init__()
        self.images = test_images
        self.current_index = 0

        self.setWindowTitle(f"Test Split Viewer ({len(test_images)} images)")
        self.setGeometry(150, 150, 1600, 900)
        self.setStyleSheet("background-color: #e9ecef;")

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Left Panel (Image) ---
        self.image_display = ImageDisplay()

        # --- Right Panel (Annotations) - Use TestAnnotationPanel ---
        self.annotation_panel = TestAnnotationPanel()

        # --- Splitter ---
        main_layout.addWidget(self.image_display, 65)
        main_layout.addWidget(self.annotation_panel, 35)  # --- Controls ---
        self.setup_controls()

        # --- Initial Load ---
        self.update_view()

    def setup_controls(self):
        controls_toolbar = self.addToolBar("Controls")
        controls_toolbar.setMovable(False)
        controls_toolbar.setStyleSheet("QToolBar { background-color: #ffc107; padding: 5px; }")  # Yellow to distinguish

        # Add label indicating this is test split
        test_label = QLabel("🧪 TEST SPLIT ONLY")
        test_label.setStyleSheet("font-weight: bold; color: #856404; padding: 0 10px;")
        controls_toolbar.addWidget(test_label)

        controls_toolbar.addSeparator()

        btn_prev = QPushButton("⬅️ Previous")
        btn_prev.clicked.connect(self.prev_image)
        controls_toolbar.addWidget(btn_prev)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("padding: 0 10px;")
        controls_toolbar.addWidget(self.status_label)

        btn_next = QPushButton("Next ➡️")
        btn_next.clicked.connect(self.next_image)
        controls_toolbar.addWidget(btn_next)

        controls_toolbar.addSeparator()

        controls_toolbar.addWidget(QLabel("Jump to Index: "))
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(80)
        self.jump_input.returnPressed.connect(self.jump_to_image)
        controls_toolbar.addWidget(self.jump_input)

        btn_zoom_in = QPushButton("➕ Zoom In")
        btn_zoom_in.clicked.connect(self.zoom_in)

        btn_zoom_out = QPushButton("➖ Zoom Out")
        btn_zoom_out.clicked.connect(self.zoom_out)

        controls_toolbar.addSeparator()
        controls_toolbar.addWidget(btn_zoom_in)
        controls_toolbar.addWidget(btn_zoom_out)

        # View mode controls
        controls_toolbar.addSeparator()
        self.btn_fit = QPushButton("Fit to Window")
        self.btn_fit.setCheckable(True)
        self.btn_fit.setChecked(False)
        self.btn_fit.toggled.connect(self.toggle_fit)
        controls_toolbar.addWidget(self.btn_fit)

        btn_reset = QPushButton("100% (1:1)")
        btn_reset.clicked.connect(self.reset_to_original)
        controls_toolbar.addWidget(btn_reset)

    def update_view(self):
        if not self.images:
            self.status_label.setText("No test images available")
            return

        image_data = self.images[self.current_index]
        self.image_display.set_image(image_data)
        self.annotation_panel.update_content(image_data)
        self.status_label.setText(f"Test Image {self.current_index + 1} / {len(self.images)}")
        self.jump_input.setText(str(self.current_index + 1))

    def prev_image(self):
        self.current_index = (self.current_index - 1 + len(self.images)) % len(self.images)
        self.update_view()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.update_view()

    def jump_to_image(self):
        try:
            idx = int(self.jump_input.text()) - 1
            if 0 <= idx < len(self.images):
                self.current_index = idx
                self.update_view()
            else:
                print(f"Index out of range. Please enter a number between 1 and {len(self.images)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def zoom_in(self):
        self.image_display.scale_factor *= 1.2
        self.image_display.update_display()

    def zoom_out(self):
        self.image_display.scale_factor /= 1.2
        self.image_display.update_display()

    def toggle_fit(self, checked: bool):
        self.image_display.set_fit_to_window(checked)
        self.image_display.update_display()

    def reset_to_original(self):
        self.image_display.reset_to_original()
        if self.btn_fit.isChecked():
            self.btn_fit.setChecked(False)


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self, images):
        super().__init__()
        self.all_images = images
        self.images = images  # Currently displayed images
        self.current_index = 0
        self.current_split = "all"  # Track current filter

        self.setWindowTitle("Annotation Visualizer")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: #e9ecef;")

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Left Panel (Image) ---
        self.image_display = ImageDisplay()

        # --- Right Panel (Annotations) ---
        self.annotation_panel = AnnotationPanel()

        # --- Splitter ---
        main_layout.addWidget(self.image_display, 65)  # 65% of space
        main_layout.addWidget(self.annotation_panel, 35)  # 35% of space

        # --- Controls ---
        self.setup_controls()

        # --- Initial Load ---
        self.update_view()

    def setup_controls(self):
        # Using a toolbar for a cleaner look
        controls_toolbar = self.addToolBar("Controls")
        controls_toolbar.setMovable(False)
        controls_toolbar.setStyleSheet("QToolBar { background-color: #dee2e6; padding: 5px; }")

        # --- Split Filter ---
        controls_toolbar.addWidget(QLabel(" Split: "))
        self.split_combo = QComboBox()
        self.split_combo.addItems(["All", "Train", "Val", "Test"])
        self.split_combo.currentTextChanged.connect(self.filter_by_split)
        controls_toolbar.addWidget(self.split_combo)

        controls_toolbar.addSeparator()

        btn_prev = QPushButton("⬅️ Previous")
        btn_prev.clicked.connect(self.prev_image)
        controls_toolbar.addWidget(btn_prev)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("padding: 0 10px;")
        controls_toolbar.addWidget(self.status_label)

        btn_next = QPushButton("Next ➡️")
        btn_next.clicked.connect(self.next_image)
        controls_toolbar.addWidget(btn_next)

        controls_toolbar.addSeparator()

        controls_toolbar.addWidget(QLabel(" Jump to Index: "))
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(80)
        self.jump_input.returnPressed.connect(self.jump_to_image)
        controls_toolbar.addWidget(self.jump_input)

        btn_zoom_in = QPushButton("➕ Zoom In")
        btn_zoom_in.clicked.connect(self.zoom_in)

        btn_zoom_out = QPushButton("➖ Zoom Out")
        btn_zoom_out.clicked.connect(self.zoom_out)

        controls_toolbar.addSeparator()
        controls_toolbar.addWidget(btn_zoom_in)
        controls_toolbar.addWidget(btn_zoom_out)

        # View mode controls
        controls_toolbar.addSeparator()
        self.btn_fit = QPushButton("Fit to Window")
        self.btn_fit.setCheckable(True)
        self.btn_fit.setChecked(False)  # default to original resolution
        self.btn_fit.toggled.connect(self.toggle_fit)
        controls_toolbar.addWidget(self.btn_fit)

        btn_reset = QPushButton("100% (1:1)")
        btn_reset.clicked.connect(self.reset_to_original)
        controls_toolbar.addWidget(btn_reset)

        # Open test split in new window
        controls_toolbar.addSeparator()
        btn_test_window = QPushButton("🪟 Open Test Split Window")
        btn_test_window.clicked.connect(self.open_test_window)
        controls_toolbar.addWidget(btn_test_window)

        self.test_window = None  # Store reference to test window

    def update_view(self):
        if not self.images:
            self.status_label.setText("No images in current filter")
            return

        image_data = self.images[self.current_index]
        self.image_display.set_image(image_data)
        self.annotation_panel.update_content(image_data)

        split_info = f" [{self.current_split.upper()}]" if self.current_split != "all" else ""
        self.status_label.setText(f"Image {self.current_index + 1} / {len(self.images)}{split_info}")
        self.jump_input.setText(str(self.current_index + 1))

    def filter_by_split(self, split_text):
        """Filter images by split (All/Train/Val/Test)."""
        split = split_text.lower()
        self.current_split = split

        if split == "all":
            self.images = self.all_images
        else:
            self.images = [img for img in self.all_images if img.get("split", "").lower() == split]

        # Reset to first image in filtered set
        self.current_index = 0

        if self.images:
            self.update_view()
        else:
            self.status_label.setText(f"No images found for split: {split_text}")
            print(f"No images found for split: {split_text}")

    def prev_image(self):
        self.current_index = (self.current_index - 1 + len(self.images)) % len(self.images)
        self.update_view()

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.update_view()

    def jump_to_image(self):
        try:
            idx = int(self.jump_input.text()) - 1
            if 0 <= idx < len(self.images):
                self.current_index = idx
                self.update_view()
            else:
                print(f"Index out of range. Please enter a number between 1 and {len(self.images)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def zoom_in(self):
        self.image_display.scale_factor *= 1.2
        self.image_display.update_display()

    def zoom_out(self):
        self.image_display.scale_factor /= 1.2
        self.image_display.update_display()

    def toggle_fit(self, checked: bool):
        self.image_display.set_fit_to_window(checked)
        # Keep status text accurate
        self.image_display.update_display()

    def reset_to_original(self):
        self.image_display.reset_to_original()
        # Also untoggle Fit button to reflect state
        if self.btn_fit.isChecked():
            self.btn_fit.setChecked(False)

    def open_test_window(self):
        """Open a separate window showing only the test split."""
        test_images = [img for img in self.all_images if img.get("split", "").lower() == "test"]

        if not test_images:
            print("No test images found!")
            self.status_label.setText("No test images found!")
            return

        # Create new window
        self.test_window = TestSplitWindow(test_images)
        self.test_window.show()
        print(f"Opened test split window with {len(test_images)} images")


def main():
    try:
        images = load_annotations()
        app = QApplication(sys.argv)

        app.setStyleSheet("""
            QPushButton {
                background-color: #007bff; color: #000000;
                border-radius: 5px; padding: 8px 12px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #0056b3; }
            QLabel { font-size: 14px; color: #212529; }
            QLineEdit { 
                padding: 8px; border: 1px solid #ced4da; 
                border-radius: 5px; font-size: 14px;
            }
        """)

        window = MainWindow(images)
        window.show()
        sys.exit(app.exec())
    except ImportError:
        print("\n--- Missing Dependencies ---")
        print("This application requires PyQt6. Please install it by running:")
        print("pip install PyQt6")
        print("--------------------------\n")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
