from __future__ import annotations

import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import yaml

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
LABEL_SUFFIXES = {".txt", ".xml"}
DEFAULT_UAV_CLASS_NAMES = ["car", "truck", "bus", "van", "Freight Car"]
DEFAULT_UAV_CLASS_ALIASES = {
    "car": ["car"],
    "truck": ["truck", "truvk"],
    "bus": ["bus"],
    "van": ["van"],
    "Freight Car": ["Freight Car", "freightcar", "feright car", "feright"],
}
DEFAULT_INVALID_CLASS_TOKENS = {"*"}


def natural_sort_key(value: str) -> list[Any]:
    """Return a stable natural-sort key for stems like 0001, frame_2, frame_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def normalize_class_key(name: str) -> str:
    """Normalize class aliases for case-insensitive matching."""
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def build_alias_map(alias_groups: dict[str, list[str]]) -> dict[str, str]:
    """Expand canonical->aliases groups into a normalized alias lookup."""
    alias_map: dict[str, str] = {}
    for canonical, aliases in alias_groups.items():
        alias_map[normalize_class_key(canonical)] = canonical
        for alias in aliases:
            alias_map[normalize_class_key(str(alias))] = canonical
    return alias_map


def build_default_uav_alias_map(class_names: list[str] | None = None) -> dict[str, str]:
    """Return the built-in DroneVehicle alias normalization map."""
    canonical_names = class_names or DEFAULT_UAV_CLASS_NAMES
    allowed_canonicals = {normalize_class_key(name) for name in canonical_names}
    alias_groups = {
        canonical: aliases
        for canonical, aliases in DEFAULT_UAV_CLASS_ALIASES.items()
        if normalize_class_key(canonical) in allowed_canonicals
    }
    return build_alias_map(alias_groups)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding and stable formatting."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML with stable ordering."""
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def relative_posix(path: Path, root: Path) -> str:
    """Return a path relative to root using forward slashes."""
    return path.relative_to(root).as_posix()


def load_image(path: Path) -> Image.Image:
    """Open an image and load pixels eagerly so the file handle can close."""
    with Image.open(path) as image:
        return image.copy()


def save_image(image: Image.Image, path: Path) -> None:
    """Save an image while preserving the destination suffix."""
    ensure_dir(path.parent)
    image.save(path)


def load_alias_map(path: Path | None) -> dict[str, str]:
    """Load alias mapping from JSON or YAML.

    Accepted formats:
    - {"car": ["Car", "CAR", "vehicle_car"]}
    - {"vehicle_car": "car", "CAR": "car"}
    """
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    alias_map: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list):
                canonical = str(key)
                alias_map[normalize_class_key(canonical)] = canonical
                for alias in value:
                    alias_map[normalize_class_key(str(alias))] = canonical
            else:
                alias_map[normalize_class_key(str(key))] = str(value)
                alias_map[normalize_class_key(str(value))] = str(value)
    return alias_map


class ClassMapper:
    """Resolve raw class tokens into stable IDs and canonical names."""

    def __init__(
        self,
        names: list[str] | None = None,
        alias_map: dict[str, str] | None = None,
        allow_new_classes: bool = True,
        invalid_tokens: set[str] | None = None,
    ) -> None:
        self.names: list[str] = []
        self.canonical_to_id: dict[str, int] = {}
        self.alias_to_canonical: dict[str, str] = {}
        self.raw_to_canonical: dict[str, str] = {}
        self.allow_new_classes = allow_new_classes
        self.invalid_class_keys = {normalize_class_key(token) for token in (invalid_tokens or set())}
        if names:
            for name in names:
                self._register_canonical(name)
        if alias_map:
            for alias, canonical in alias_map.items():
                self.alias_to_canonical[alias] = canonical
                self._register_canonical(canonical)

    def _register_canonical(self, canonical: str) -> int:
        normalized = normalize_class_key(canonical)
        if normalized in self.canonical_to_id:
            return self.canonical_to_id[normalized]
        class_id = len(self.names)
        self.names.append(canonical)
        self.canonical_to_id[normalized] = class_id
        self.alias_to_canonical[normalized] = canonical
        return class_id

    def resolve(self, raw_class: str) -> tuple[int, str]:
        token = str(raw_class).strip()
        if re.fullmatch(r"[+-]?\d+", token):
            class_id = int(token)
            if class_id < 0:
                raise ValueError(f"Negative class id '{token}' is not allowed.")
            if not self.allow_new_classes and class_id >= len(self.names):
                raise ValueError(f"Unknown class id '{token}' is outside the configured class range.")
            while len(self.names) <= class_id:
                self._register_canonical(str(len(self.names)))
            canonical = self.names[class_id]
            self.raw_to_canonical[token] = canonical
            return class_id, canonical
        normalized = normalize_class_key(token)
        if not normalized:
            raise ValueError("Blank class name is not allowed.")
        if normalized in self.invalid_class_keys:
            raise ValueError(f"Invalid placeholder class '{token}'.")
        canonical = self.alias_to_canonical.get(normalized)
        if canonical is None:
            if not self.allow_new_classes:
                raise ValueError(f"Unknown class '{token}' is not in the configured class map.")
            canonical = token
        class_id = self._register_canonical(canonical)
        self.alias_to_canonical[normalized] = canonical
        self.raw_to_canonical[token] = canonical
        return class_id, canonical

    def export(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping report."""
        return {
            "names": self.names,
            "canonical_to_id": {
                canonical: self.canonical_to_id[normalize_class_key(canonical)] for canonical in self.names
            },
            "aliases": dict(sorted(self.alias_to_canonical.items())),
            "raw_to_canonical": dict(sorted(self.raw_to_canonical.items())),
        }


@dataclass
class OBBObject:
    """Internal absolute-pixel OBB representation."""

    class_id: int
    class_name: str
    polygon: np.ndarray
    source: str
    raw_class: str
    meta: dict[str, Any] = field(default_factory=dict)

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x0, y0, x1, y1) for the polygon."""
        x_coords = self.polygon[:, 0]
        y_coords = self.polygon[:, 1]
        return float(x_coords.min()), float(y_coords.min()), float(x_coords.max()), float(y_coords.max())

    def normalized_polygon(self, width: int, height: int) -> np.ndarray:
        """Return polygon coordinates normalized to [0, 1]."""
        normalized = self.polygon.copy().astype(np.float32)
        normalized[:, 0] /= max(width, 1)
        normalized[:, 1] /= max(height, 1)
        return normalized

    def to_label_line(self, width: int, height: int) -> str:
        """Return one training label line in polygon format."""
        normalized = np.clip(self.normalized_polygon(width, height), 0.0, 1.0)
        flat = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        return f"{self.class_id} {flat}"


@dataclass
class RawPair:
    """One RGB/IR pair before preprocessing."""

    split: str
    key: str
    rgb_image: Path
    ir_image: Path
    rgb_label: Path | None = None
    ir_label: Path | None = None


def build_pair_key(stem: str, remove_tokens: list[str] | None = None) -> str:
    """Normalize a file stem into a pairing key."""
    key = stem
    for token in remove_tokens or []:
        if token:
            key = key.replace(token, "")
    return key


def scan_files_by_key(root: Path, suffixes: set[str], remove_tokens: list[str] | None = None) -> dict[str, list[Path]]:
    """Collect files keyed by normalized stem."""
    if not root.exists():
        return {}
    indexed: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*"), key=lambda current: natural_sort_key(current.stem)):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = build_pair_key(path.stem, remove_tokens)
            indexed.setdefault(key, []).append(path)
    return indexed


def pick_preferred_label(paths: list[Path]) -> Path:
    """Prefer TXT labels over XML when both exist for the same key."""
    sorted_paths = sorted(paths, key=lambda item: (item.suffix.lower() != ".txt", item.name.lower()))
    return sorted_paths[0]


def find_label_path(indexed: dict[str, list[Path]], key: str) -> Path | None:
    """Select one label file path for a key."""
    paths = indexed.get(key)
    if not paths:
        return None
    return pick_preferred_label(paths)


def clamp_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp polygon coordinates to the image extent."""
    clamped = points.copy().astype(np.float32)
    clamped[:, 0] = np.clip(clamped[:, 0], 0.0, max(width - 1.0, 0.0))
    clamped[:, 1] = np.clip(clamped[:, 1], 0.0, max(height - 1.0, 0.0))
    return clamped


def polygon_area(points: np.ndarray) -> float:
    """Return polygon area using the shoelace formula."""
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return float(abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))) * 0.5)


def min_area_rect_points(points: np.ndarray) -> np.ndarray:
    """Convert any point set to a stable 4-point rectangle."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    return rotate_polygon_start(order_polygon_ccw(box.astype(np.float32)))


def order_polygon_ccw(points: np.ndarray) -> np.ndarray:
    """Sort polygon points counter-clockwise around the centroid."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def rotate_polygon_start(points: np.ndarray) -> np.ndarray:
    """Rotate points so the first point is the top-most, then left-most."""
    order = np.lexsort((points[:, 0], points[:, 1]))
    start = int(order[0])
    return np.roll(points, -start, axis=0).astype(np.float32)


def stable_rectangle(points: np.ndarray) -> np.ndarray:
    """Return a stable 4-point rectangle representation."""
    return rotate_polygon_start(order_polygon_ccw(min_area_rect_points(points)))


def polygon_from_xywha(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle: float,
    angle_unit: str = "auto",
) -> np.ndarray:
    """Convert xywha to a 4-point polygon."""
    radians = angle
    if angle_unit == "deg":
        radians = math.radians(angle)
    elif angle_unit == "auto" and abs(angle) > math.tau + 1e-3:
        radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    half_w = width / 2.0
    half_h = height / 2.0
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = corners @ rotation.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotate_polygon_start(order_polygon_ccw(rotated))


def is_normalized(values: np.ndarray) -> bool:
    """Return True if coordinates look normalized."""
    return float(values.max()) <= 1.01 and float(values.min()) >= -0.01


def resolve_polygon(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Resolve polygon coordinates to absolute pixels and a stable 4-point rectangle."""
    polygon = points.astype(np.float32).reshape(-1, 2)
    if is_normalized(polygon):
        polygon[:, 0] *= image_width
        polygon[:, 1] *= image_height
    if len(polygon) != 4:
        polygon = min_area_rect_points(polygon)
    return clamp_polygon(stable_rectangle(polygon), image_width, image_height)


def resolve_xywha(values: np.ndarray, image_width: int, image_height: int, angle_unit: str) -> np.ndarray:
    """Resolve xywha coordinates to a stable absolute polygon."""
    center_x, center_y, box_width, box_height, angle = values.astype(np.float32).tolist()
    if max(abs(center_x), abs(center_y), abs(box_width), abs(box_height)) <= 1.01:
        center_x *= image_width
        center_y *= image_height
        box_width *= image_width
        box_height *= image_height
    polygon = polygon_from_xywha(center_x, center_y, box_width, box_height, angle, angle_unit=angle_unit)
    return clamp_polygon(stable_rectangle(polygon), image_width, image_height)


def parse_text_labels(
    path: Path,
    image_width: int,
    image_height: int,
    class_mapper: ClassMapper,
    angle_unit: str,
    source_name: str,
) -> tuple[list[OBBObject], list[str]]:
    """Parse TXT labels that are xywha or polygon-based."""
    issues: list[str] = []
    objects: list[OBBObject] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in {6, 9} and not (len(parts) > 9 and (len(parts) - 1) % 2 == 0):
            issues.append(f"{path.name}:{line_number} unsupported column count {len(parts)}")
            continue
        try:
            class_id, class_name = class_mapper.resolve(parts[0])
        except ValueError as error:
            issues.append(f"{path.name}:{line_number} ignored object with class '{parts[0]}': {error}")
            continue
        numeric = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        if len(parts) == 6:
            polygon = resolve_xywha(numeric, image_width, image_height, angle_unit)
        else:
            polygon = resolve_polygon(numeric, image_width, image_height)
        objects.append(
            OBBObject(
                class_id=class_id,
                class_name=class_name,
                polygon=polygon,
                source=source_name,
                raw_class=parts[0],
                meta={"label_file": str(path)},
            )
        )
    return objects, issues


def _find_first_text(element: ET.Element, tags: tuple[str, ...]) -> str | None:
    for tag in tags:
        node = element.find(tag)
        if node is not None and node.text is not None:
            return node.text.strip()
    return None


def parse_xml_labels(
    path: Path,
    image_width: int,
    image_height: int,
    class_mapper: ClassMapper,
    angle_unit: str,
    source_name: str,
) -> tuple[list[OBBObject], list[str]]:
    """Parse XML labels with polygon, robndbox, or axis-aligned bndbox objects."""
    issues: list[str] = []
    objects: list[OBBObject] = []
    tree = ET.parse(path)
    root = tree.getroot()
    for index, obj in enumerate(root.findall(".//object"), start=1):
        raw_class = _find_first_text(obj, ("name", "label", "class")) or "0"
        try:
            class_id, class_name = class_mapper.resolve(raw_class)
        except ValueError as error:
            issues.append(f"{path.name}:object_{index} ignored object with class '{raw_class}': {error}")
            continue
        polygon: np.ndarray | None = None

        polygon_node = obj.find("polygon")
        if polygon_node is not None:
            points: list[float] = []
            for tag in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"):
                node = polygon_node.find(tag)
                if node is not None and node.text is not None:
                    points.append(float(node.text))
            if not points:
                for point_node in polygon_node.findall("pt"):
                    x_text = _find_first_text(point_node, ("x",))
                    y_text = _find_first_text(point_node, ("y",))
                    if x_text is not None and y_text is not None:
                        points.extend([float(x_text), float(y_text)])
            if len(points) >= 8 and len(points) % 2 == 0:
                polygon = resolve_polygon(np.asarray(points, dtype=np.float32), image_width, image_height)

        if polygon is None:
            robndbox = obj.find("robndbox") or obj.find("rotated_bndbox") or obj.find("rbox")
            if robndbox is not None:
                center_x = float(_find_first_text(robndbox, ("cx", "x_center", "xcenter")) or 0.0)
                center_y = float(_find_first_text(robndbox, ("cy", "y_center", "ycenter")) or 0.0)
                box_width = float(_find_first_text(robndbox, ("w", "width")) or 0.0)
                box_height = float(_find_first_text(robndbox, ("h", "height")) or 0.0)
                angle = float(_find_first_text(robndbox, ("angle", "theta", "rotation")) or 0.0)
                polygon = resolve_xywha(
                    np.asarray([center_x, center_y, box_width, box_height, angle], dtype=np.float32),
                    image_width,
                    image_height,
                    angle_unit,
                )

        if polygon is None:
            bbox = obj.find("bndbox")
            if bbox is not None:
                xmin = float(_find_first_text(bbox, ("xmin", "x1", "left")) or 0.0)
                ymin = float(_find_first_text(bbox, ("ymin", "y1", "top")) or 0.0)
                xmax = float(_find_first_text(bbox, ("xmax", "x2", "right")) or 0.0)
                ymax = float(_find_first_text(bbox, ("ymax", "y2", "bottom")) or 0.0)
                polygon = resolve_polygon(
                    np.asarray([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], dtype=np.float32),
                    image_width,
                    image_height,
                )

        if polygon is None:
            issues.append(f"{path.name}:object_{index} has no supported geometry")
            continue

        objects.append(
            OBBObject(
                class_id=class_id,
                class_name=class_name,
                polygon=polygon,
                source=source_name,
                raw_class=raw_class,
                meta={"label_file": str(path)},
            )
        )
    return objects, issues


def parse_label_file(
    path: Path | None,
    image_width: int,
    image_height: int,
    class_mapper: ClassMapper,
    angle_unit: str,
    source_name: str,
) -> tuple[list[OBBObject], list[str]]:
    """Parse one label file into internal OBB objects."""
    if path is None or not path.exists():
        return [], []
    if path.suffix.lower() == ".txt":
        return parse_text_labels(path, image_width, image_height, class_mapper, angle_unit, source_name)
    if path.suffix.lower() == ".xml":
        return parse_xml_labels(path, image_width, image_height, class_mapper, angle_unit, source_name)
    return [], [f"{path.name} unsupported label suffix '{path.suffix}'"]


def union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return the union of two xyxy boxes."""
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def detect_valid_bbox(image: Image.Image, white_thresh: int) -> tuple[int, int, int, int]:
    """Detect the non-white content region."""
    array = np.asarray(image)
    if array.ndim == 2:
        valid_mask = array < white_thresh
    else:
        valid_mask = np.any(array < white_thresh, axis=2)
    if not np.any(valid_mask):
        return 0, 0, image.width, image.height
    coords = np.argwhere(valid_mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def apply_target_protection(
    crop_box: tuple[int, int, int, int],
    objects: list[OBBObject],
    image_width: int,
    image_height: int,
    protect_size: int,
) -> tuple[int, int, int, int]:
    """Expand a crop box if a target is too close to an edge."""
    half = max(protect_size / 2.0, 1.0)
    x0, y0, x1, y1 = crop_box
    changed = True
    while changed:
        changed = False
        for obj in objects:
            obj_x0, obj_y0, obj_x1, obj_y1 = obj.bbox()
            center_x = (obj_x0 + obj_x1) / 2.0
            center_y = (obj_y0 + obj_y1) / 2.0
            protect_x0 = int(math.floor(min(obj_x0, center_x - half)))
            protect_y0 = int(math.floor(min(obj_y0, center_y - half)))
            protect_x1 = int(math.ceil(max(obj_x1, center_x + half)))
            protect_y1 = int(math.ceil(max(obj_y1, center_y + half)))
            new_box = (x0, y0, x1, y1)
            if obj_x0 - x0 < half:
                new_box = (min(x0, protect_x0), new_box[1], new_box[2], new_box[3])
            if obj_y0 - y0 < half:
                new_box = (new_box[0], min(y0, protect_y0), new_box[2], new_box[3])
            if x1 - obj_x1 < half:
                new_box = (new_box[0], new_box[1], max(x1, protect_x1), new_box[3])
            if y1 - obj_y1 < half:
                new_box = (new_box[0], new_box[1], new_box[2], max(y1, protect_y1))
            clamped = (
                max(0, new_box[0]),
                max(0, new_box[1]),
                min(image_width, new_box[2]),
                min(image_height, new_box[3]),
            )
            if clamped != (x0, y0, x1, y1):
                x0, y0, x1, y1 = clamped
                changed = True
    return x0, y0, x1, y1


def clip_polygon_to_box(points: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Clip a polygon to an axis-aligned crop box using Sutherland-Hodgman."""
    x_min, y_min, x_max, y_max = crop_box
    polygon = points.astype(np.float32)

    def clip_edge(vertices: list[np.ndarray], inside_fn, intersect_fn) -> list[np.ndarray]:
        if not vertices:
            return []
        output: list[np.ndarray] = []
        previous = vertices[-1]
        previous_inside = inside_fn(previous)
        for current in vertices:
            current_inside = inside_fn(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect_fn(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect_fn(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def intersect_vertical(a: np.ndarray, b: np.ndarray, x_value: float) -> np.ndarray:
        if abs(b[0] - a[0]) < 1e-6:
            return np.array([x_value, a[1]], dtype=np.float32)
        ratio = (x_value - a[0]) / (b[0] - a[0])
        y_value = a[1] + ratio * (b[1] - a[1])
        return np.array([x_value, y_value], dtype=np.float32)

    def intersect_horizontal(a: np.ndarray, b: np.ndarray, y_value: float) -> np.ndarray:
        if abs(b[1] - a[1]) < 1e-6:
            return np.array([a[0], y_value], dtype=np.float32)
        ratio = (y_value - a[1]) / (b[1] - a[1])
        x_value = a[0] + ratio * (b[0] - a[0])
        return np.array([x_value, y_value], dtype=np.float32)

    vertices = [point.astype(np.float32) for point in polygon]
    vertices = clip_edge(vertices, lambda p: p[0] >= x_min, lambda a, b: intersect_vertical(a, b, float(x_min)))
    vertices = clip_edge(vertices, lambda p: p[0] <= x_max, lambda a, b: intersect_vertical(a, b, float(x_max)))
    vertices = clip_edge(vertices, lambda p: p[1] >= y_min, lambda a, b: intersect_horizontal(a, b, float(y_min)))
    vertices = clip_edge(vertices, lambda p: p[1] <= y_max, lambda a, b: intersect_horizontal(a, b, float(y_max)))
    if len(vertices) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)


def crop_images_and_labels(
    rgb_image: Image.Image,
    ir_image: Image.Image,
    objects: list[OBBObject],
    white_thresh: int,
    protect_size: int,
    min_area: float,
) -> tuple[Image.Image, Image.Image, list[OBBObject], tuple[int, int, int, int], list[str]]:
    """Crop two aligned modalities and move labels into the cropped frame."""
    if rgb_image.size != ir_image.size:
        raise ValueError(f"RGB/IR size mismatch: {rgb_image.size} vs {ir_image.size}")
    image_width, image_height = rgb_image.size
    crop_box = union_box(detect_valid_bbox(rgb_image, white_thresh), detect_valid_bbox(ir_image, white_thresh))
    crop_box = apply_target_protection(crop_box, objects, image_width, image_height, protect_size)
    crop_events: list[str] = []

    rgb_cropped = rgb_image.crop(crop_box)
    ir_cropped = ir_image.crop(crop_box)

    kept_objects: list[OBBObject] = []
    x0, y0, x1, y1 = crop_box
    cropped_width, cropped_height = rgb_cropped.size
    for obj in objects:
        clipped = clip_polygon_to_box(obj.polygon, crop_box)
        if len(clipped) == 0:
            crop_events.append(f"dropped:{obj.class_name}:{obj.source}:outside_crop")
            continue
        rect = stable_rectangle(clipped)
        shifted = rect.copy()
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
        shifted = clamp_polygon(shifted, cropped_width, cropped_height)
        if polygon_area(shifted) < min_area:
            crop_events.append(f"dropped:{obj.class_name}:{obj.source}:degenerate_after_crop")
            continue
        kept_objects.append(
            OBBObject(
                class_id=obj.class_id,
                class_name=obj.class_name,
                polygon=shifted,
                source=obj.source,
                raw_class=obj.raw_class,
                meta={**obj.meta, "crop_box": [x0, y0, x1, y1]},
            )
        )
    return rgb_cropped, ir_cropped, kept_objects, crop_box, crop_events


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU for two convex polygons."""
    a_poly = stable_rectangle(a).astype(np.float32)
    b_poly = stable_rectangle(b).astype(np.float32)
    area_a = polygon_area(a_poly)
    area_b = polygon_area(b_poly)
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(a_poly, b_poly)
    union_area = area_a + area_b - float(intersection_area)
    if union_area <= 1e-6:
        return 0.0
    return float(intersection_area) / union_area


def merge_two_objects(rgb_obj: OBBObject, ir_obj: OBBObject) -> OBBObject:
    """Merge two aligned detections into one OBB by fitting a joint min-area rectangle."""
    merged_polygon = stable_rectangle(np.vstack([rgb_obj.polygon, ir_obj.polygon]))
    return OBBObject(
        class_id=rgb_obj.class_id,
        class_name=rgb_obj.class_name,
        polygon=merged_polygon,
        source="fused",
        raw_class=rgb_obj.raw_class,
        meta={"rgb_source": rgb_obj.source, "ir_source": ir_obj.source},
    )


def fuse_label_sets(
    rgb_objects: list[OBBObject],
    ir_objects: list[OBBObject],
    iou_threshold: float,
) -> tuple[list[OBBObject], dict[str, int], list[str]]:
    """Fuse RGB and IR labels conservatively.

    Matching rule:
    - same class id,
    - IoU >= threshold,
    - unmatched objects are retained and logged instead of overwritten.
    """
    if not rgb_objects and not ir_objects:
        return [], {"matched": 0, "rgb_only": 0, "ir_only": 0, "class_conflict": 0}, []
    if not rgb_objects:
        return ir_objects, {"matched": 0, "rgb_only": 0, "ir_only": len(ir_objects), "class_conflict": 0}, []
    if not ir_objects:
        return rgb_objects, {"matched": 0, "rgb_only": len(rgb_objects), "ir_only": 0, "class_conflict": 0}, []

    fused: list[OBBObject] = []
    events: list[str] = []
    used_rgb: set[int] = set()
    used_ir: set[int] = set()
    stats = {"matched": 0, "rgb_only": 0, "ir_only": 0, "class_conflict": 0}

    candidates: list[tuple[float, int, int]] = []
    for rgb_index, rgb_obj in enumerate(rgb_objects):
        for ir_index, ir_obj in enumerate(ir_objects):
            iou = polygon_iou(rgb_obj.polygon, ir_obj.polygon)
            if rgb_obj.class_id == ir_obj.class_id and iou >= iou_threshold:
                candidates.append((iou, rgb_index, ir_index))
            elif rgb_obj.class_id != ir_obj.class_id and iou >= iou_threshold:
                stats["class_conflict"] += 1
                events.append(
                    f"class_conflict:{rgb_obj.class_name}:{ir_obj.class_name}:iou={iou:.4f}:rgb={rgb_index}:ir={ir_index}"
                )
    for _, rgb_index, ir_index in sorted(candidates, key=lambda item: item[0], reverse=True):
        if rgb_index in used_rgb or ir_index in used_ir:
            continue
        used_rgb.add(rgb_index)
        used_ir.add(ir_index)
        fused.append(merge_two_objects(rgb_objects[rgb_index], ir_objects[ir_index]))
        stats["matched"] += 1

    for index, obj in enumerate(rgb_objects):
        if index not in used_rgb:
            fused.append(obj)
            stats["rgb_only"] += 1
            events.append(f"rgb_only:{obj.class_name}:{index}")
    for index, obj in enumerate(ir_objects):
        if index not in used_ir:
            fused.append(obj)
            stats["ir_only"] += 1
            events.append(f"ir_only:{obj.class_name}:{index}")
    return fused, stats, events


def write_label_file(objects: list[OBBObject], path: Path, width: int, height: int) -> None:
    """Write polygon labels in current Ultralytics-compatible OBB format."""
    ensure_dir(path.parent)
    lines = [obj.to_label_line(width, height) for obj in sorted(objects, key=lambda item: (item.class_id, item.class_name))]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def copy_text_file(source: Path, destination: Path) -> None:
    """Copy a small text file."""
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)


def validate_polygon_line(line: str, num_classes: int | None = None) -> tuple[bool, str | None]:
    """Validate one normalized polygon label line."""
    parts = line.strip().split()
    if not parts:
        return True, None
    if len(parts) != 9:
        return False, f"expected 9 columns, got {len(parts)}"
    try:
        class_id = int(parts[0])
    except ValueError:
        return False, f"class id '{parts[0]}' is not an integer"
    if class_id < 0:
        return False, "class id must be non-negative"
    if num_classes is not None and class_id >= num_classes:
        return False, f"class id {class_id} exceeds num_classes {num_classes}"
    coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)
    if float(coords.min()) < -0.01 or float(coords.max()) > 1.01:
        return False, "polygon coordinates are out of normalized bounds"
    if polygon_area(coords) <= 1e-6:
        return False, "polygon area is degenerate"
    return True, None


def stems_from_directory(root: Path) -> dict[str, Path]:
    """Index files by stem for checker scripts."""
    indexed: dict[str, Path] = {}
    if not root.exists():
        return indexed
    for path in root.rglob("*"):
        if path.is_file():
            indexed[path.stem] = path
    return indexed
