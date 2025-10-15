#!/usr/bin/env python3
"""TAIGA IACT likelihood estimation utilities.

This module offers a compact template-based likelihood estimator that can
operate directly on the event dumps contained in this repository.  The
original CTA ``ImPACT`` implementation depends on heavy external
packages (``ctapipe``, large template libraries, ...).  For the TAIGA
use-case we provide a pure-Python alternative that only relies on the
standard library.  The code is intentionally lightweight to keep the
dependencies manageable while still offering a practical baseline for
parameter estimation.

Workflow summary
================

* Parse the ``*_clean_*.txt`` file to obtain sparse camera images.
* Parse the matching ``*_hillas_*.csv`` file to obtain the ground truth
  parameters.
* Rotate the ground impact point and arrival azimuth into the telescope
  reference frame to remove the dependence on the azimuthal pointing
  angle.
* Build a :class:`TemplateLibrary` that stores the cleaned images and the
  associated parameters.
* Use inverse distance weighting in the parameter space to interpolate an
  expected image for arbitrary parameter values.
* Compare the interpolated template with the observation under a
  Gaussian noise model to obtain a log-likelihood.  A crude stochastic
  optimiser searches for the parameter set that maximises that
  likelihood.

The module is designed to be driven through configuration files instead
of a command line interface.  See :func:`run_reconstruction_from_config`
for the high level entry point used in the repository notebooks.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "PixelMeasurement",
    "EventImage",
    "EventParameters",
    "TAIGADataset",
    "TemplateLibrary",
    "LikelihoodModel",
    "estimate_event_parameters",
    "load_configuration",
    "run_reconstruction_from_config",
]

Vector = List[float]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelMeasurement:
    """Single cleaned pixel measurement."""

    cluster_id: int
    pixel_id: int
    x: float
    y: float
    amplitude: float


@dataclass
class EventImage:
    """Collection of pixel measurements belonging to one event."""

    header: Vector
    pixels: List[PixelMeasurement]
    _vector_cache: Dict[int, Vector] = None  # filled lazily

    def to_vector(self, indexer: "PixelIndexer") -> Vector:
        cache_key = id(indexer.lookup)
        if self._vector_cache is None:
            self._vector_cache = {}
        if cache_key not in self._vector_cache:
            vector = [0.0] * indexer.size
            for pixel in self.pixels:
                key = (pixel.cluster_id, pixel.pixel_id)
                idx = indexer.lookup.get(key)
                if idx is not None:
                    vector[idx] = pixel.amplitude
            self._vector_cache[cache_key] = vector
        stored = self._vector_cache[cache_key]
        return list(stored)


@dataclass(frozen=True)
class EventParameters:
    """Ground truth parameters associated with one event."""

    x_ground: float
    y_ground: float
    energy: float
    xmax: float
    source_tet: float
    tel_fi: float
    source_fi: float

    def local_ground_coordinates(self) -> Tuple[float, float]:
        """Return the ground impact point in the telescope frame."""

        cos_fi = math.cos(self.tel_fi)
        sin_fi = math.sin(self.tel_fi)
        x_local = self.x_ground * cos_fi + self.y_ground * sin_fi
        y_local = -self.x_ground * sin_fi + self.y_ground * cos_fi
        return x_local, y_local

    def local_source_fi(self) -> float:
        """Return the arrival azimuth in the telescope frame."""

        return wrap_angle(self.source_fi - self.tel_fi)

    def template_vector(self) -> Vector:
        local_x, local_y = self.local_ground_coordinates()
        return [
            float(local_x),
            float(local_y),
            float(self.energy),
            float(self.xmax),
            float(self.source_tet),
            float(self.local_source_fi()),
        ]

    @staticmethod
    def ground_from_local(
        x_local: float, y_local: float, tel_fi: float
    ) -> Tuple[float, float]:
        cos_fi = math.cos(tel_fi)
        sin_fi = math.sin(tel_fi)
        x_global = x_local * cos_fi - y_local * sin_fi
        y_global = x_local * sin_fi + y_local * cos_fi
        return x_global, y_global

    @staticmethod
    def source_fi_from_local(local_source_fi: float, tel_fi: float) -> float:
        return wrap_angle(local_source_fi + tel_fi)


def wrap_angle(value: float) -> float:
    """Map angles to the interval ``[-pi, pi)``."""

    value = (value + math.pi) % (2.0 * math.pi)
    return value - math.pi


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


def _parse_event_header(header_line: str) -> Tuple[Vector, int]:
    values = [float(x) for x in header_line.split()]
    if len(values) < 4:
        raise ValueError(
            f"Malformed event header (expected >=4 values): {header_line!r}"
        )
    pixel_count = int(values[3])
    if pixel_count < 0:
        raise ValueError(f"Negative pixel count encountered: {pixel_count}")
    return values, pixel_count


def _parse_pixel_measurement(line: str) -> PixelMeasurement:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            "Malformed pixel line: expected five entries (cluster, pixel, x, y,"
            f" amplitude) but got {len(parts)} in line {line!r}"
        )
    return PixelMeasurement(
        cluster_id=int(parts[0]),
        pixel_id=int(parts[1]),
        x=float(parts[2]),
        y=float(parts[3]),
        amplitude=float(parts[4]),
    )


def read_event_images(path: Path) -> List[EventImage]:
    """Parse a ``*_clean_*.txt`` file into :class:`EventImage` objects."""

    events: List[EventImage] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    cursor = 0
    total_lines = len(lines)
    while cursor < total_lines:
        header, n_pixels = _parse_event_header(lines[cursor])
        cursor += 1
        if cursor + n_pixels > total_lines:
            raise ValueError(
                "Unexpected end of file when reading pixel data for an event"
            )
        pixels = [
            _parse_pixel_measurement(lines[cursor + offset])
            for offset in range(n_pixels)
        ]
        events.append(EventImage(header=header, pixels=pixels))
        cursor += n_pixels

    return events


def read_event_parameters(path: Path) -> List[EventParameters]:
    """Parse the matching ``*_hillas_*.csv`` file."""

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = [
            "x_ground",
            "y_ground",
            "energy",
            "Xmax",
            "source_tet",
            "tel_fi",
            "source_fi",
        ]
        for column in required:
            if column not in reader.fieldnames:
                raise ValueError(
                    f"CSV file is missing required column {column!r}"
                )
        params: List[EventParameters] = []
        for row in reader:
            params.append(
                EventParameters(
                    x_ground=float(row["x_ground"]),
                    y_ground=float(row["y_ground"]),
                    energy=float(row["energy"]),
                    xmax=float(row["Xmax"]),
                    source_tet=float(row["source_tet"]),
                    tel_fi=float(row["tel_fi"]),
                    source_fi=float(row["source_fi"]),
                )
            )
    return params


# ---------------------------------------------------------------------------
# Dataset representation
# ---------------------------------------------------------------------------


class PixelIndexer:
    """Utility that maps ``(cluster_id, pixel_id)`` pairs to indices."""

    def __init__(self, events: Sequence[EventImage]):
        lookup: Dict[Tuple[int, int], int] = {}
        positions: List[Tuple[float, float]] = []
        for event in events:
            for pixel in event.pixels:
                key = (pixel.cluster_id, pixel.pixel_id)
                if key not in lookup:
                    lookup[key] = len(lookup)
                    positions.append((pixel.x, pixel.y))
        self.lookup = lookup
        self.positions = positions
        self.size = len(lookup)


class TAIGADataset:
    """Container that bundles event images with their parameters."""

    def __init__(self, txt_path: Path, csv_path: Path):
        if not txt_path.exists():
            raise FileNotFoundError(f"Event text file not found: {txt_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Parameter csv file not found: {csv_path}")

        events = read_event_images(txt_path)
        parameters = read_event_parameters(csv_path)
        if len(events) != len(parameters):
            raise ValueError(
                f"Text / CSV event count mismatch: {len(events)} vs {len(parameters)}"
            )

        self.events = events
        self.parameters = parameters
        self.template_vectors = [params.template_vector() for params in parameters]
        self.pixel_indexer = PixelIndexer(events)

    def __len__(self) -> int:
        return len(self.events)

    def leave_one_out(self, index: int) -> "TAIGADataset":
        if index < 0 or index >= len(self.events):
            raise IndexError(f"Event index {index} out of range")
        dataset = TAIGADataset.__new__(TAIGADataset)  # type: ignore[misc]
        dataset.events = [
            event for idx, event in enumerate(self.events) if idx != index
        ]
        dataset.parameters = [
            params for idx, params in enumerate(self.parameters) if idx != index
        ]
        dataset.template_vectors = [p.template_vector() for p in dataset.parameters]
        dataset.pixel_indexer = PixelIndexer(dataset.events)
        return dataset


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


def _vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(component * component for component in values))


class TemplateLibrary:
    """Interpolate between discrete template images using IDW."""

    def __init__(
        self,
        dataset: TAIGADataset,
        idw_power: float = 2.0,
        max_templates: int = 64,
    ):
        if len(dataset) == 0:
            raise ValueError("Cannot build a template library from an empty dataset")
        self.dataset = dataset
        self.idw_power = float(idw_power)
        self.max_templates = int(max(1, max_templates))
        self.indexer = dataset.pixel_indexer

        self.templates: List[Vector] = [
            event.to_vector(self.indexer) for event in dataset.events
        ]
        self.parameters: List[Vector] = [
            vector[:] for vector in dataset.template_vectors
        ]

        # Parameter-wise scale factors to normalise the distance metric.
        minima = self.parameters[0][:]
        maxima = self.parameters[0][:]
        for param_vector in self.parameters[1:]:
            for idx, value in enumerate(param_vector):
                minima[idx] = min(minima[idx], value)
                maxima[idx] = max(maxima[idx], value)
        scales: List[float] = []
        for idx in range(len(minima)):
            if idx == 5:
                scales.append(math.pi)
            else:
                scales.append(max(maxima[idx] - minima[idx], 1.0))
        self.parameter_scales = scales

    def interpolate(self, params: Vector) -> Tuple[Vector, Vector]:
        if len(params) != len(self.parameter_scales):
            raise ValueError(
                "Parameter vector must contain six entries:"
                " (x_local, y_local, energy, Xmax, source_tet, source_fi_local)"
            )

        # Compute distances to all templates and retain only the closest ones.
        import heapq

        candidates: List[Tuple[float, int]] = []
        for idx, vector in enumerate(self.parameters):
            deltas = []
            for axis, component in enumerate(vector):
                scale = self.parameter_scales[axis]
                if axis == 5:
                    delta = wrap_angle(component - params[axis]) / scale
                else:
                    delta = (component - params[axis]) / scale
                deltas.append(delta)
            distance = max(_vector_norm(deltas), 1e-6)
            candidates.append((distance, idx))

        nearest = heapq.nsmallest(self.max_templates, candidates, key=lambda item: item[0])

        weights: List[float] = []
        template_indices: List[int] = []
        for distance, index in nearest:
            weight = 1.0 / (distance ** self.idw_power)
            weights.append(weight)
            template_indices.append(index)

        total_weight = sum(weights)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("Invalid weight normalisation encountered")
        normalised = [w / total_weight for w in weights]

        template = [0.0] * self.indexer.size
        for weight, index in zip(normalised, template_indices):
            image = self.templates[index]
            for idx, amplitude in enumerate(image):
                template[idx] += weight * amplitude

        return template, normalised


# ---------------------------------------------------------------------------
# Likelihood evaluation
# ---------------------------------------------------------------------------


class LikelihoodModel:
    """Gaussian log-likelihood between an observation and interpolated template."""

    def __init__(
        self,
        library: TemplateLibrary,
        noise_floor: float = 4.0,
        pedestal_variance: float = 9.0,
    ) -> None:
        self.library = library
        self.noise_floor = max(noise_floor, 1e-3)
        self.pedestal_variance = max(pedestal_variance, 1e-6)

    def expected_image(self, params: Vector) -> Vector:
        template, _ = self.library.interpolate(params[:])
        return template

    def log_likelihood(self, event: EventImage, params: Vector) -> float:
        observed = event.to_vector(self.library.indexer)
        predicted = self.expected_image(params)

        total = 0.0
        noise_floor_sq = self.noise_floor ** 2
        for obs, pred in zip(observed, predicted):
            variance = max(pred + self.pedestal_variance, noise_floor_sq)
            residual = obs - pred
            total += (residual * residual) / variance + math.log(2.0 * math.pi * variance)
        return -0.5 * total

    def maximise(
        self,
        event: EventImage,
        bounds: Sequence[Tuple[float, float]],
        rng: random.Random,
        n_samples: int = 4096,
        refine_iterations: int = 4,
        refine_radius: float = 0.1,
    ) -> Tuple[Vector, float]:
        if len(bounds) != 6:
            raise ValueError("Bounds must define six (min, max) pairs")
        lows = [float(lo) for lo, _ in bounds]
        highs = [float(hi) for _, hi in bounds]
        spans = [hi - lo for lo, hi in bounds]
        for span in spans:
            if span <= 0:
                raise ValueError("Invalid parameter bounds")

        def random_sample(scale: Sequence[float]) -> Vector:
            return [lo + rng.random() * span for lo, span in zip(lows, scale)]

        best_params = random_sample(spans)
        best_ll = self.log_likelihood(event, best_params)

        for _ in range(max(1, int(n_samples))):
            candidate = random_sample(spans)
            ll = self.log_likelihood(event, candidate)
            if ll > best_ll:
                best_ll = ll
                best_params = candidate

        step = [span * refine_radius for span in spans]
        for _ in range(refine_iterations):
            for _ in range(128):
                proposal = [
                    max(lo, min(hi, rng.gauss(mu, sigma)))
                    for mu, sigma, lo, hi in zip(best_params, step, lows, highs)
                ]
                ll = self.log_likelihood(event, proposal)
                if ll > best_ll:
                    best_ll = ll
                    best_params = proposal
            step = [sigma * 0.5 for sigma in step]

        return best_params, best_ll


# ---------------------------------------------------------------------------
# High level helper
# ---------------------------------------------------------------------------


def _default_bounds(dataset: TAIGADataset) -> List[Tuple[float, float]]:
    params = [vector[:] for vector in dataset.template_vectors]
    minima = params[0][:]
    maxima = params[0][:]
    for vector in params[1:]:
        for idx, value in enumerate(vector):
            minima[idx] = min(minima[idx], value)
            maxima[idx] = max(maxima[idx], value)

    margins = [100.0, 100.0, 10.0, 50.0, math.radians(1.0), math.radians(5.0)]
    minima = [m - margin for m, margin in zip(minima, margins)]
    maxima = [m + margin for m, margin in zip(maxima, margins)]

    minima[0] = max(minima[0], -1000.0)
    minima[1] = max(minima[1], -1000.0)
    maxima[0] = min(maxima[0], 1000.0)
    maxima[1] = min(maxima[1], 1000.0)
    minima[2] = max(minima[2], 5.0)
    maxima[2] = min(maxima[2], 400.0)
    minima[3] = max(minima[3], 100.0)
    maxima[3] = min(maxima[3], 900.0)
    minima[4] = max(minima[4], math.radians(25.0))
    maxima[4] = min(maxima[4], math.radians(45.0))
    minima[5] = max(minima[5], -math.pi)
    maxima[5] = min(maxima[5], math.pi)

    return list(zip(minima, maxima))


def estimate_event_parameters(
    dataset: TAIGADataset,
    event_index: int,
    *,
    leave_one_out: bool = True,
    rng: Optional[random.Random] = None,
    n_samples: int = 4096,
) -> Tuple[Vector, float, EventParameters]:
    if rng is None:
        rng = random.Random()
    if leave_one_out and len(dataset) > 1:
        working_set = dataset.leave_one_out(event_index)
    else:
        working_set = dataset

    library = TemplateLibrary(working_set)
    model = LikelihoodModel(library)

    bounds = _default_bounds(dataset)
    best_params, best_ll = model.maximise(
        dataset.events[event_index],
        bounds=bounds,
        rng=rng,
        n_samples=n_samples,
    )

    true_params = dataset.parameters[event_index]
    return best_params, best_ll, true_params


def _matching_csv_path(txt_path: Path) -> Path:
    stem = txt_path.name
    if "_clean_" not in stem:
        raise ValueError(
            f"Cannot derive CSV filename from {txt_path!s}: missing '_clean_' marker"
        )
    csv_name = stem.replace("_clean_", "_hillas_")
    csv_name = csv_name.rsplit(".", 1)[0] + ".csv"
    return txt_path.with_name(csv_name)


def _load_datasets(
    paths: Sequence[Path], limit: Optional[int] = None
) -> TAIGADataset:
    paths = list(paths)
    if not paths:
        raise ValueError("At least one template file must be provided")

    events: List[EventImage] = []
    parameters: List[EventParameters] = []
    for txt_path in paths:
        csv_path = _matching_csv_path(txt_path)
        dataset = TAIGADataset(txt_path, csv_path)
        events.extend(dataset.events)
        parameters.extend(dataset.parameters)

    if limit is not None:
        limit = max(0, int(limit))
        events = events[:limit]
        parameters = parameters[:limit]

    combined = TAIGADataset.__new__(TAIGADataset)  # type: ignore[misc]
    combined.events = events
    combined.parameters = parameters
    combined.template_vectors = [p.template_vector() for p in parameters]
    combined.pixel_indexer = PixelIndexer(events)
    return combined


def load_configuration(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = ["template_txt_files", "test_txt_file", "output_csv"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Configuration missing required key {key!r}")
    return payload


def _iter_template_paths(entries: Iterable[str]) -> Iterator[Path]:
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        yield path


def run_reconstruction_from_config(config_path: Path) -> List[Dict[str, float]]:
    config = load_configuration(config_path)
    template_paths = list(
        _iter_template_paths(config["template_txt_files"])  # type: ignore[arg-type]
    )
    template_limit = config.get("template_event_limit")
    template_dataset = _load_datasets(template_paths, limit=template_limit)

    library = TemplateLibrary(
        template_dataset,
        idw_power=float(config.get("idw_power", 2.0)),
        max_templates=int(config.get("max_templates", 64)),
    )
    model = LikelihoodModel(
        library,
        noise_floor=float(config.get("noise_floor", 4.0)),
        pedestal_variance=float(config.get("pedestal_variance", 9.0)),
    )
    bounds = _default_bounds(template_dataset)

    test_txt = Path(config["test_txt_file"]).expanduser().resolve()
    test_csv = _matching_csv_path(test_txt)
    test_dataset = TAIGADataset(test_txt, test_csv)

    seed = config.get("random_seed")
    rng = random.Random(seed)
    n_samples = int(config.get("n_samples", 4096))
    refine_iterations = int(config.get("refine_iterations", 4))
    refine_radius = float(config.get("refine_radius", 0.1))
    test_limit = config.get("test_event_limit")

    results: List[Dict[str, float]] = []
    for index, (event, truth) in enumerate(zip(test_dataset.events, test_dataset.parameters)):
        if test_limit is not None and index >= int(test_limit):
            break
        best_params_local, best_ll = model.maximise(
            event,
            bounds=bounds,
            rng=rng,
            n_samples=n_samples,
            refine_iterations=refine_iterations,
            refine_radius=refine_radius,
        )

        x_local, y_local, energy, xmax, source_tet, source_fi_local = best_params_local
        x_global, y_global = EventParameters.ground_from_local(x_local, y_local, truth.tel_fi)
        source_fi_global = EventParameters.source_fi_from_local(
            source_fi_local, truth.tel_fi
        )

        results.append(
            {
                "event_index": float(index),
                "log_likelihood": float(best_ll),
                "x_ground_est": float(x_global),
                "y_ground_est": float(y_global),
                "energy_est": float(energy),
                "xmax_est": float(xmax),
                "source_tet_est": float(source_tet),
                "source_fi_est": float(source_fi_global),
                "x_ground_true": float(truth.x_ground),
                "y_ground_true": float(truth.y_ground),
                "energy_true": float(truth.energy),
                "xmax_true": float(truth.xmax),
                "source_tet_true": float(truth.source_tet),
                "source_fi_true": float(truth.source_fi),
                "tel_fi": float(truth.tel_fi),
            }
        )

    output_csv = Path(config["output_csv"]).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_index",
                "log_likelihood",
                "x_ground_est",
                "y_ground_est",
                "energy_est",
                "xmax_est",
                "source_tet_est",
                "source_fi_est",
                "x_ground_true",
                "y_ground_true",
                "energy_true",
                "xmax_true",
                "source_tet_true",
                "source_fi_true",
                "tel_fi",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results
