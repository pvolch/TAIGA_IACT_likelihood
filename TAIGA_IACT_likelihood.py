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
  parameters ``(x_ground, y_ground, energy, Xmax, source_tet)``.
* Build a :class:`TemplateLibrary` that stores the cleaned images and the
  associated parameters.
* Use inverse distance weighting in the parameter space to interpolate an
  expected image for arbitrary parameter values.
* Compare the interpolated template with the observation under a
  Gaussian noise model to obtain a log-likelihood.  A crude stochastic
  optimiser searches for the parameter set that maximises that
  likelihood.

The implementation exposes both a Python API and a small command line
interface.  Run ``python TAIGA_IACT_likelihood.py --help`` for an
overview of the available options.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PixelMeasurement",
    "EventImage",
    "EventParameters",
    "TAIGADataset",
    "TemplateLibrary",
    "LikelihoodModel",
    "estimate_event_parameters",
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
                idx = indexer.lookup[key]
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

    def as_list(self) -> Vector:
        return [
            float(self.x_ground),
            float(self.y_ground),
            float(self.energy),
            float(self.xmax),
            float(self.source_tet),
        ]


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
        required = ["x_ground", "y_ground", "energy", "Xmax", "source_tet"]
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
            params.as_list() for params in dataset.parameters
        ]

        # Parameter-wise scale factors to normalise the distance metric.
        minima = self.parameters[0][:]
        maxima = self.parameters[0][:]
        for param_vector in self.parameters[1:]:
            for idx, value in enumerate(param_vector):
                minima[idx] = min(minima[idx], value)
                maxima[idx] = max(maxima[idx], value)
        self.parameter_scales = [
            max(maxima[idx] - minima[idx], 1.0) for idx in range(5)
        ]

    def interpolate(self, params: Vector) -> Tuple[Vector, Vector]:
        if len(params) != 5:
            raise ValueError(
                "Parameter vector must contain five entries:"
                " (x_ground, y_ground, energy, Xmax, source_tet)"
            )

        # Compute distances to all templates and retain only the closest ones.
        import heapq

        candidates: List[Tuple[float, int]] = []
        for idx, vector in enumerate(self.parameters):
            deltas = [
                (component - params[idx]) / self.parameter_scales[idx]
                for idx, component in enumerate(vector)
            ]
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
        if len(bounds) != 5:
            raise ValueError("Bounds must define five (min, max) pairs")
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
    params = [p.as_list() for p in dataset.parameters]
    minima = params[0][:]
    maxima = params[0][:]
    for vector in params[1:]:
        for idx, value in enumerate(vector):
            minima[idx] = min(minima[idx], value)
            maxima[idx] = max(maxima[idx], value)

    margins = [100.0, 100.0, 10.0, 50.0, math.radians(1.0)]
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

    return list(zip(minima, maxima))


def estimate_event_parameters(
    dataset: TAIGADataset,
    event_index: int,
    *,
    leave_one_out: bool = True,
    rng: Optional[random.Random] = None,
    n_samples: int = 4096,
) -> Tuple[Vector, float, Vector]:
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

    true_params = dataset.parameters[event_index].as_list()
    return best_params, best_ll, true_params


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _format_params(params: Sequence[float]) -> str:
    return (
        "x_ground={:.2f} m, y_ground={:.2f} m, energy={:.2f} TeV, "
        "Xmax={:.2f} g/cm^2, source_tet={:.2f} rad"
    ).format(*params)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Likelihood-based parameter estimation for TAIGA events",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=Path("data/taiga607_clean_iact01_14_7fix_cb0.txt"),
        help="Path to the cleaned event dump (*.txt)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/taiga607_hillas_iact01_14_7fix_cb0.csv"),
        help="Path to the matching parameter table (*.csv)",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="Index of the event to reconstruct",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="Number of random samples used during optimisation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the internal random number generator",
    )
    parser.add_argument(
        "--keep-template",
        action="store_true",
        help="Do not remove the target event from the template library",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    dataset = TAIGADataset(args.txt, args.csv)
    if args.event_index < 0 or args.event_index >= len(dataset):
        raise SystemExit(
            f"Event index {args.event_index} out of range (0..{len(dataset)-1})"
        )

    rng = random.Random(args.seed)
    best_params, best_ll, true_params = estimate_event_parameters(
        dataset,
        event_index=args.event_index,
        leave_one_out=not args.keep_template,
        rng=rng,
        n_samples=args.samples,
    )

    print("Selected event:", args.event_index)
    print("True parameters:", _format_params(true_params))
    print("Estimated parameters:", _format_params(best_params))
    print("Log-likelihood at optimum: {:.3f}".format(best_ll))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
