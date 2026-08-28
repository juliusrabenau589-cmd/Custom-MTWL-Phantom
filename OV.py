from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

from pylinac.core.geometry import Point
from pylinac.core.scale import MachineScale, convert
from pylinac.metrics.image import GlobalSizedFieldLocator
from pylinac.winston_lutz import (
    BBConfig,
    WinstonLutzMultiTargetMultiField,
    WinstonLutzMultiTargetMultiFieldImage,
)


# ---------------- Einstellungen ----------------

DICOM_DIR = Path(
    r"C:\Users\juliu\Desktop\Julius\Uni\Master MP"
    r"\Proekt\Dicom\MM\05.26\Dicom_neu"
)

# PDF und TXT werden direkt im Ordner der DICOM-Bilder gespeichert.
TXT_FILE = DICOM_DIR / "Oliver_Hough_BB_Mittelpunkte.txt"
PDF_FILE = DICOM_DIR / "Oliver_Hough_Auswertung.pdf"

BB_ORDER = ["Iso", "1", "2", "3", "4"]

EXTRA_BUFFER_PERCENT = 40.0
UPSAMPLE_FACTOR = 5.0

BB_R_MIN_FRACTION = 0.85
BB_R_MAX_FRACTION = 0.95

HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 8
HOUGH_THRESHOLD_START = 100
HOUGH_THRESHOLD_MIN = 1

BB_COLOUR_PERCENTILE = 25.0
BKGD_COLOUR_PERCENTILE = 75.0

BLUR_KSIZE = 3
BLUR_SIGMA = 2

BB_PROXIMITY_MM = 10.0
BB_CENTER_SEARCH_RADIUS_MM = 4.0

# Nur Darstellung
EDGE_TOLERANCE_UPSAMPLED_PX = 2.0
FIELD_EDGE_POINT_SPACING_MM = 2.0


# ---------------- Hilfsfunktionen ----------------

def format_decimal(value):
    return f"{value:.4f}".replace(".", ",")


def linear_upsample(array, factor):
    """Lineares Upsampling auf dem von Oliver verwendeten Abtastraster."""
    rows = np.arange(array.shape[0], dtype=float)
    columns = np.arange(array.shape[1], dtype=float)

    interpolator = RegularGridInterpolator(
        (rows, columns),
        array,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    new_rows = np.arange(
        0,
        array.shape[0],
        1.0 / factor,
    )
    new_columns = np.arange(
        0,
        array.shape[1],
        1.0 / factor,
    )

    rr, cc = np.meshgrid(
        new_rows,
        new_columns,
        indexing="ij",
    )

    points = np.column_stack(
        (rr.ravel(), cc.ravel())
    )

    return interpolator(points).reshape(
        len(new_rows),
        len(new_columns),
    )


def bb_magnification(image, bb_config):
    """BB-spezifischer geometrischer Vergrößerungsfaktor."""
    bb_position = np.array(
        [
            bb_config.offset_up_mm,
            bb_config.offset_left_mm,
            bb_config.offset_in_mm,
        ],
        dtype=float,
    )

    gantry_iec, _, couch_iec = convert(
        input_scale=image.machine_scale,
        output_scale=MachineScale.IEC61217,
        gantry=image.gantry_angle,
        collimator=0,
        rotation=image.couch_angle,
    )

    rotated_position = Rotation.from_euler(
        "xyz",
        [-couch_iec, 0, gantry_iec],
        degrees=True,
    ).apply(bb_position)

    denominator = image.sad - rotated_position[0]

    if denominator <= 0:
        raise ValueError(
            "Ungültige BB-Geometrie: Abstand zur Quelle <= 0."
        )

    return float(image.sad / denominator)


def circle_mask(shape, center_x, center_y, radius):
    """Boolesche Kreismaske im vollständigen Bildarray."""
    yy, xx = np.ogrid[
        :shape[0],
        :shape[1],
    ]

    return (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )


def extract_display_edges(
    image_8bit,
    center_x,
    center_y,
    radius,
):
    """
    Canny-Kantenpixel nahe dem akzeptierten Hough-Kreis.
    Diese Funktion beeinflusst die Erkennung nicht.
    """
    edges = cv.Canny(
        image_8bit,
        threshold1=max(int(HOUGH_PARAM1 / 2), 1),
        threshold2=max(int(HOUGH_PARAM1), 1),
    )

    yy, xx = np.mgrid[
        :image_8bit.shape[0],
        :image_8bit.shape[1],
    ]

    near_circle = (
        np.abs(
            np.hypot(
                xx - center_x,
                yy - center_y,
            )
            - radius
        )
        <= EDGE_TOLERANCE_UPSAMPLED_PX
    )

    edge_y, edge_x = np.nonzero(
        (edges > 0) & near_circle
    )

    return (
        edge_x.astype(float),
        edge_y.astype(float),
    )


# ---------------- pylinac-Felder ----------------

def locate_pylinac_fields(image):
    """
    Wiederholt pylinacs GlobalSizedFieldLocator mit denselben
    Feldgrößenparametern wie find_field_centroids().

    Zurückgegeben werden je BB-Name:
    - pylinac-Feldmittelpunkt
    - pylinac-Feldgrenze
    - aus der Grenze gefüllte Feldmaske
    """
    field_sizes = [
        bb.rad_size_mm
        for bb in image.bb_arrangement
    ]

    mean_size = (
        max(field_sizes)
        + min(field_sizes)
    ) / 2.0

    tolerance = max(
        (
            max(field_sizes)
            - min(field_sizes)
        ) * 1.2,
        0.1 * mean_size,
    )

    locator = GlobalSizedFieldLocator.from_physical(
        max_number=len(image.bb_arrangement),
        field_height_mm=mean_size,
        field_width_mm=mean_size,
        field_tolerance_mm=tolerance,
    )

    detected_centers = image.compute(
        metrics=locator
    )

    matched_centers = image.find_field_matches(
        detected_centers,
        bb_proximity_mm=BB_PROXIMITY_MM,
    )

    center_boundary_pairs = list(
        zip(
            locator.fields,
            locator.boundaries,
        )
    )

    used_indices = set()
    fields = {}

    for bb_name, matched_center in matched_centers.items():
        available = [
            index
            for index in range(
                len(center_boundary_pairs)
            )
            if index not in used_indices
        ]

        if not available:
            continue

        index = min(
            available,
            key=lambda i: (
                center_boundary_pairs[i][0]
                .distance_to(matched_center)
            ),
        )

        used_indices.add(index)

        boundary = (
            center_boundary_pairs[index][1]
            .astype(bool)
        )

        fields[bb_name] = {
            "center": matched_center,
            "boundary": boundary,
            "mask": ndimage.binary_fill_holes(
                boundary
            ),
        }

    return fields


def ordered_contour(mask):
    """Geordnet umlaufende Außenkontur einer pylinac-Feldmaske."""
    contours, _ = cv.findContours(
        mask.astype(np.uint8) * 255,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE,
    )

    if not contours:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    contour = max(
        contours,
        key=cv.contourArea,
    )

    points = contour[
        :, 0, :
    ].astype(float)

    return (
        points[:, 0],
        points[:, 1],
    )


# ---------------- Oliver-Hough-BB-Erkennung ----------------

def detect_bb_hough(
    roi_image,
    field_mask_roi,
    bb_radius_mm,
    original_dpmm,
    magnification,
    expected_x_up,
    expected_y_up,
):
    """
    Oliver-basierte Hough-Erkennung.

    Die zusätzliche Oliver-Feldsegmentierung wurde entfernt.
    Die Feldmaske für den Hintergrundvergleich stammt von pylinac.
    """
    image = roi_image.astype(float).copy()

    # Grounding und Normalisierung
    image -= image.min()

    if image.max() <= 0:
        raise ValueError(
            "Das ROI besitzt keinen Intensitätsbereich."
        )

    image /= image.max()

    # GaussianBlur
    blur_size = int(BLUR_KSIZE)

    if blur_size % 2 == 0:
        blur_size += 1

    image = cv.GaussianBlur(
        image,
        ksize=(blur_size, blur_size),
        sigmaX=float(BLUR_SIGMA),
        sigmaY=float(BLUR_SIGMA),
    )

    # Upsampling
    image = linear_upsample(
        image,
        UPSAMPLE_FACTOR,
    )

    image_8bit = np.round(
        image * 255.0 / image.max()
    ).astype(np.uint8)

    # pylinac-Feldmaske auf dieselbe hochskalierte Größe bringen
    field_mask_up = cv.resize(
        field_mask_roi.astype(np.uint8),
        (
            image_8bit.shape[1],
            image_8bit.shape[0],
        ),
        interpolation=cv.INTER_NEAREST,
    ).astype(bool)

    upsampled_dpmm = (
        original_dpmm
        * UPSAMPLE_FACTOR
    )

    expected_radius = (
        bb_radius_mm
        * upsampled_dpmm
        * magnification
    )

    minimum_radius = max(
        int(
            np.round(
                expected_radius
                * BB_R_MIN_FRACTION
            )
        ),
        1,
    )

    maximum_radius = max(
        int(
            np.round(
                expected_radius
                * BB_R_MAX_FRACTION
            )
        ),
        minimum_radius,
    )

    search_radius_up = (
        BB_CENTER_SEARCH_RADIUS_MM
        * upsampled_dpmm
    )

    hough_margin = int(
        np.ceil(
            search_radius_up
            + maximum_radius
            + 2
        )
    )

    hough_x_min = max(
        int(np.floor(expected_x_up))
        - hough_margin,
        0,
    )
    hough_x_max = min(
        int(np.ceil(expected_x_up))
        + hough_margin
        + 1,
        image_8bit.shape[1],
    )
    hough_y_min = max(
        int(np.floor(expected_y_up))
        - hough_margin,
        0,
    )
    hough_y_max = min(
        int(np.ceil(expected_y_up))
        + hough_margin
        + 1,
        image_8bit.shape[0],
    )

    if (
        hough_x_min >= hough_x_max
        or hough_y_min >= hough_y_max
    ):
        raise ValueError(
            "Die erwartete BB-Position liegt außerhalb des ROI."
        )

    hough_image = image_8bit[
        hough_y_min:hough_y_max,
        hough_x_min:hough_x_max,
    ]

    for threshold in range(
        HOUGH_THRESHOLD_START,
        HOUGH_THRESHOLD_MIN - 1,
        -1,
    ):
        circles = cv.HoughCircles(
            hough_image,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=(
                HOUGH_MIN_DIST
                * UPSAMPLE_FACTOR
            ),
            param1=HOUGH_PARAM1,
            param2=threshold,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )

        if circles is None:
            continue

        candidates = []

        for circle in circles[0]:
            x, y, radius = (
                circle.astype(np.int32)
            )

            x += hough_x_min
            y += hough_y_min

            distance_to_expected = np.hypot(
                x - expected_x_up,
                y - expected_y_up,
            )

            if (
                distance_to_expected
                > search_radius_up
            ):
                continue

            inner_mask = circle_mask(
                image_8bit.shape,
                x,
                y,
                max(int(radius * 0.8), 1),
            )

            values = image_8bit[
                inner_mask
            ]
            values = values[
                values > 0
            ]

            if values.size == 0:
                continue

            candidates.append(
                (
                    float(values.mean()),
                    int(x),
                    int(y),
                    int(radius),
                )
            )

        # Oliver: dunkelste Kreise zuerst prüfen
        for _, x, y, radius in sorted(
            candidates,
            key=lambda item: item[0],
        ):
            inner_mask = circle_mask(
                image_8bit.shape,
                x,
                y,
                max(int(radius * 0.8), 1),
            )

            bb_values = image_8bit[
                inner_mask
            ]
            bb_values = bb_values[
                bb_values > 0
            ]

            excluded_bb = circle_mask(
                image_8bit.shape,
                x,
                y,
                radius,
            )

            background_values = image_8bit[
                field_mask_up
                & (~excluded_bb)
            ]
            background_values = (
                background_values[
                    background_values > 0
                ]
            )

            if (
                bb_values.size == 0
                or background_values.size == 0
            ):
                continue

            bb_value = np.percentile(
                bb_values,
                BB_COLOUR_PERCENTILE,
            )

            background_value = np.percentile(
                background_values,
                BKGD_COLOUR_PERCENTILE,
            )

            if bb_value >= background_value:
                continue

            edge_x, edge_y = (
                extract_display_edges(
                    image_8bit,
                    x,
                    y,
                    radius,
                )
            )

            return {
                "x_up": float(x),
                "y_up": float(y),
                "radius_up": float(radius),
                "edge_x_up": edge_x,
                "edge_y_up": edge_y,
            }

    raise ValueError(
        "Kein akzeptierter lokaler Hough-Kreis gefunden."
    )


# ---------------- Integration in pylinac ----------------

def hough_find_bb_centroids(
    self,
    bb_diameter_mm,
    low_density=False,
):
    """
    Ersetzt nur die BB-Lokalisierung von pylinac.
    Felder, Zuordnung und spätere Abstandsberechnung bleiben pylinac.
    """
    if low_density:
        raise ValueError(
            "Die Methode setzt dunkle, hochdichte BBs voraus."
        )

    fields = locate_pylinac_fields(self)

    points = []
    self.hough_candidates = []

    for bb in self.bb_arrangement:
        if bb.name not in fields:
            continue

        field_center = fields[bb.name][
            "center"
        ]
        field_mask = fields[bb.name][
            "mask"
        ]

        roi_size_px = (
            bb.rad_size_mm
            * (
                1.0
                + EXTRA_BUFFER_PERCENT / 100.0
            )
            * self.dpmm
        )

        x_start = max(
            int(
                np.round(
                    field_center.x
                    - roi_size_px / 2.0
                )
            ),
            0,
        )
        x_stop = min(
            int(
                np.round(
                    field_center.x
                    + roi_size_px / 2.0
                )
            ),
            self.array.shape[1] - 1,
        )

        y_start = max(
            int(
                np.round(
                    field_center.y
                    - roi_size_px / 2.0
                )
            ) - 1,
            0,
        )
        y_stop = min(
            int(
                np.round(
                    field_center.y
                    + roi_size_px / 2.0
                )
            ) + 1,
            self.array.shape[0] - 1,
        )

        roi = self.array[
            y_start:y_stop + 1,
            x_start:x_stop + 1,
        ]

        field_mask_roi = field_mask[
            y_start:y_stop + 1,
            x_start:x_stop + 1,
        ]

        expected_bb = self.nominal_bb_position(
            bb
        )

        try:
            result = detect_bb_hough(
                roi_image=roi,
                field_mask_roi=field_mask_roi,
                bb_radius_mm=bb.bb_size_mm / 2.0,
                original_dpmm=self.dpmm,
                magnification=bb_magnification(
                    self,
                    bb,
                ),
                expected_x_up=(
                    expected_bb.x - x_start
                ) * UPSAMPLE_FACTOR,
                expected_y_up=(
                    expected_bb.y - y_start
                ) * UPSAMPLE_FACTOR,
            )
        except ValueError as error:
            raise ValueError(
                f"{Path(self.path).name}, "
                f"BB {bb.name}: {error}"
            ) from error

        point = Point(
            x=(
                x_start
                + result["x_up"]
                / UPSAMPLE_FACTOR
            ),
            y=(
                y_start
                + result["y_up"]
                / UPSAMPLE_FACTOR
            ),
        )

        points.append(point)

        contour_x, contour_y = (
            ordered_contour(field_mask)
        )

        self.hough_candidates.append(
            {
                "bb_name": bb.name,
                "point": point,
                "radius_px": (
                    result["radius_up"]
                    / UPSAMPLE_FACTOR
                ),
                "bb_edge_x_px": (
                    x_start
                    + result["edge_x_up"]
                    / UPSAMPLE_FACTOR
                ),
                "bb_edge_y_px": (
                    y_start
                    + result["edge_y_up"]
                    / UPSAMPLE_FACTOR
                ),
                "field_center": field_center,
                "field_contour_x_px": contour_x,
                "field_contour_y_px": contour_y,
                "roi": (
                    x_start,
                    x_stop,
                    y_start,
                    y_stop,
                ),
                "dpmm": float(self.dpmm),
            }
        )

    return points


WinstonLutzMultiTargetMultiFieldImage.find_bb_centroids = (
    hough_find_bb_centroids
)


# ---------------- PDF-Hilfsfunktionen ----------------

def candidate_for_bb(image, bb_name):
    return next(
        (
            candidate
            for candidate in getattr(
                image,
                "hough_candidates",
                [],
            )
            if candidate["bb_name"] == bb_name
        ),
        None,
    )


def sample_contour(
    x,
    y,
    spacing_px,
):
    """Punkte mit ungefähr konstantem Abstand entlang einer Kontur."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or spacing_px <= 0:
        return x, y

    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    segment_lengths = np.hypot(
        np.diff(x_closed),
        np.diff(y_closed),
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(segment_lengths))
    )

    if cumulative[-1] <= 0:
        return x[:1], y[:1]

    distances = np.arange(
        0.0,
        cumulative[-1],
        spacing_px,
    )

    return (
        np.interp(
            distances,
            cumulative,
            x_closed,
        ),
        np.interp(
            distances,
            cumulative,
            y_closed,
        ),
    )


def add_overlay(
    ax,
    candidate,
    x_offset=0.0,
    y_offset=0.0,
    show_label=True,
):
    """BB- und Feldmarkierungen in ein Achsenobjekt einzeichnen."""
    ax.scatter(
        candidate["bb_edge_x_px"]
        - x_offset,
        candidate["bb_edge_y_px"]
        - y_offset,
        c="red",
        marker="s",
        s=4,
        linewidths=0,
        alpha=0.6,
        zorder=5,
    )

    ax.plot(
        candidate["point"].x
        - x_offset,
        candidate["point"].y
        - y_offset,
        marker="o",
        color="red",
        markersize=3,
        linestyle="None",
        zorder=6,
    )

    ax.plot(
        candidate["field_center"].x
        - x_offset,
        candidate["field_center"].y
        - y_offset,
        marker="+",
        color="lime",
        markersize=7,
        markeredgewidth=1.2,
        linestyle="None",
        zorder=7,
    )

    contour_x, contour_y = sample_contour(
        candidate["field_contour_x_px"],
        candidate["field_contour_y_px"],
        FIELD_EDGE_POINT_SPACING_MM
        * candidate["dpmm"],
    )

    ax.scatter(
        contour_x - x_offset,
        contour_y - y_offset,
        c="yellow",
        marker="o",
        s=7,
        linewidths=0,
        alpha=0.8,
        zorder=4,
    )

    if show_label:
        ax.text(
            candidate["point"].x
            - x_offset
            + 3,
            candidate["point"].y
            - y_offset
            - 3,
            f"BB {candidate['bb_name']}",
            fontsize=7,
            color="white",
            bbox={
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
                "pad": 1.5,
            },
            zorder=8,
        )


def epid_iso_metrics(image):
    """
    EPID-Mitte, Pixelwert dort und Abstand zur pylinac-Mitte
    des Iso-Feldes.
    """
    epid = image.epid

    pixel_value = float(
        ndimage.map_coordinates(
            image.array.astype(float),
            np.array(
                [
                    [epid.y],
                    [epid.x],
                ]
            ),
            order=1,
            mode="nearest",
        )[0]
    )

    metrics = {
        "epid_x": float(epid.x),
        "epid_y": float(epid.y),
        "pixel_value": pixel_value,
        "iso_field_x": None,
        "iso_field_y": None,
        "distance_px": None,
        "distance_mm": None,
    }

    iso_match = image.arrangement_matches.get(
        "Iso"
    )

    if iso_match is not None:
        distance_px = epid.distance_to(
            iso_match.field
        )

        metrics.update(
            {
                "iso_field_x": float(
                    iso_match.field.x
                ),
                "iso_field_y": float(
                    iso_match.field.y
                ),
                "distance_px": float(
                    distance_px
                ),
                "distance_mm": float(
                    distance_px / image.dpmm
                ),
            }
        )

    return metrics


def full_page_zoom_limits(
    image,
    margin_px=35,
):
    candidates = getattr(
        image,
        "hough_candidates",
        [],
    )

    if not candidates:
        return (
            0,
            image.array.shape[1] - 1,
            0,
            image.array.shape[0] - 1,
        )

    x_values = []
    y_values = []

    for candidate in candidates:
        x_values.extend(
            candidate[
                "field_contour_x_px"
            ].tolist()
        )
        y_values.extend(
            candidate[
                "field_contour_y_px"
            ].tolist()
        )

    return (
        max(
            int(np.floor(min(x_values) - margin_px)),
            0,
        ),
        min(
            int(np.ceil(max(x_values) + margin_px)),
            image.array.shape[1] - 1,
        ),
        max(
            int(np.floor(min(y_values) - margin_px)),
            0,
        ),
        min(
            int(np.ceil(max(y_values) + margin_px)),
            image.array.shape[0] - 1,
        ),
    )


def plot_full_image_page(
    fig,
    image,
):
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.2, 1.3],
    )

    ax_image = fig.add_subplot(
        grid[0, 0]
    )
    ax_text = fig.add_subplot(
        grid[0, 1]
    )

    ax_image.imshow(
        image.array,
        cmap="gray",
        origin="upper",
    )

    for candidate in getattr(
        image,
        "hough_candidates",
        [],
    ):
        add_overlay(
            ax_image,
            candidate,
        )

    x_min, x_max, y_min, y_max = (
        full_page_zoom_limits(image)
    )

    # pylinac-EPID-Mitte
    ax_image.axvline(
        image.epid.x,
        color="deepskyblue",
        linewidth=0.7,
        alpha=0.6,
    )
    ax_image.axhline(
        image.epid.y,
        color="deepskyblue",
        linewidth=0.7,
        alpha=0.6,
    )

    ax_image.set_xlim(
        x_min,
        x_max,
    )
    ax_image.set_ylim(
        y_max,
        y_min,
    )

    ax_image.set_title(
        (
            f"{Path(image.path).name}\n"
            f"Gantry={image.gantry_angle:.0f}°, "
            f"Kollimator={image.collimator_angle:.0f}°, "
            f"Tisch={image.couch_angle:.0f}°"
        ),
        fontsize=10,
    )

    ax_image.set_xlabel("x [px]")
    ax_image.set_ylabel("y [px]")
    ax_image.tick_params(
        axis="both",
        labelsize=8,
    )

    metrics = epid_iso_metrics(image)

    lines = [
        Path(image.path).name,
        "",
        f"Gantry: {image.gantry_angle:.0f}°",
        f"Kollimator: {image.collimator_angle:.0f}°",
        f"Tisch: {image.couch_angle:.0f}°",
        "",
        "EPID-Mitte (pylinac)",
        f"  x_px: {format_decimal(metrics['epid_x'])}",
        f"  y_px: {format_decimal(metrics['epid_y'])}",
        "  Pixelwert: "
        f"{format_decimal(metrics['pixel_value'])}",
        "",
    ]

    if metrics["iso_field_x"] is not None:
        lines.extend(
            [
                "Iso-Feldmitte (pylinac)",
                "  x_px: "
                f"{format_decimal(metrics['iso_field_x'])}",
                "  y_px: "
                f"{format_decimal(metrics['iso_field_y'])}",
                "  Abstand EPID–Iso-Feld:",
                "    px: "
                f"{format_decimal(metrics['distance_px'])}",
                "    mm: "
                f"{format_decimal(metrics['distance_mm'])}",
                "",
            ]
        )

    for bb_name in BB_ORDER:
        match = image.arrangement_matches.get(
            bb_name
        )

        if match is None:
            lines.extend(
                [
                    f"BB {bb_name}",
                    "  nicht erkannt",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                f"BB {bb_name}",
                f"  x_px: {format_decimal(match.bb.x)}",
                f"  y_px: {format_decimal(match.bb.y)}",
                "  BB_Feld_2D_mm: "
                f"{format_decimal(match.bb_field_distance_mm)}",
                "",
            ]
        )

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        family="monospace",
    )


def plot_bb_crop(
    ax,
    image,
    candidate,
):
    (
        x_start,
        x_stop,
        y_start,
        y_stop,
    ) = candidate["roi"]

    ax.imshow(
        image.array[
            y_start:y_stop + 1,
            x_start:x_stop + 1,
        ],
        cmap="gray",
        origin="upper",
    )

    add_overlay(
        ax,
        candidate,
        x_offset=x_start,
        y_offset=y_start,
        show_label=False,
    )

    ax.text(
        0.03,
        0.97,
        f"BB {candidate['bb_name']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={
            "facecolor": "black",
            "alpha": 0.45,
            "edgecolor": "none",
            "pad": 2,
        },
    )

    ax.set_title(
        (
            f"G{image.gantry_angle:.0f}  "
            f"K{image.collimator_angle:.0f}  "
            f"T{image.couch_angle:.0f}"
        ),
        fontsize=8,
    )

    ax.set_xticks([])
    ax.set_yticks([])


def write_custom_pdf(
    wl,
    filename,
):
    images = sorted(
        wl.images,
        key=lambda image: Path(
            image.path
        ).name,
    )

    with PdfPages(filename) as pdf:
        # Gesamtergebnisse
        fig = plt.figure(
            figsize=(11.69, 8.27),
        )
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            wl.results(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
        )
        fig.suptitle(
            "Winston-Lutz Multi-Target Auswertung",
            fontsize=13,
        )
        fig.tight_layout(
            rect=[0, 0, 1, 0.96]
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Große Einzelbildseiten
        for image in images:
            fig = plt.figure(
                figsize=(11.69, 8.27),
            )
            plot_full_image_page(
                fig,
                image,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 4x4-Raster pro BB
        images_per_page = 16

        for bb_name in BB_ORDER:
            for start in range(
                0,
                len(images),
                images_per_page,
            ):
                chunk = images[
                    start:
                    start + images_per_page
                ]

                fig, axes = plt.subplots(
                    4,
                    4,
                    figsize=(11.69, 8.27),
                )

                axes = axes.ravel()

                for ax in axes:
                    ax.axis("off")

                for ax, image in zip(
                    axes,
                    chunk,
                ):
                    ax.axis("on")

                    candidate = candidate_for_bb(
                        image,
                        bb_name,
                    )

                    if candidate is None:
                        ax.text(
                            0.5,
                            0.5,
                            (
                                f"BB {bb_name}\n"
                                "nicht erkannt"
                            ),
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    plot_bb_crop(
                        ax,
                        image,
                        candidate,
                    )

                fig.suptitle(
                    f"BB {bb_name}",
                    fontsize=12,
                )
                fig.tight_layout(
                    rect=[0, 0, 1, 0.95]
                )
                pdf.savefig(fig)
                plt.close(fig)


# ---------------- Textdatei ----------------

def write_results(
    wl,
    filename,
):
    """Schreibt die rekonstruierten 3D-Targetmittelpunkte und wl.results() in eine TXT-Datei."""

    results_text = wl.results()

    with open(
        filename,
        "w",
        encoding="utf-8-sig",
    ) as file:

        file.write("Winston-Lutz Multi-Target Auswertung\n")
        file.write("===================================\n\n")

        # ---------------------------------
        # Ermittelte 3D-Targetmittelpunkte
        # ---------------------------------

        file.write("Ermittelte Targetmittelpunkte\n")
        file.write("-----------------------------\n\n")

        file.write(
            f"{'Target':<10}"
            f"{'X [mm]':>12}"
            f"{'Y [mm]':>12}"
            f"{'Z [mm]':>12}\n"
        )

        file.write("-" * 46 + "\n")

        for bb in wl.bbs:
            position = bb.measured_bb_position
            name = bb.bb_config.name

            file.write(
                f"{name:<10}"
                f"{position.x:>12.3f}"
                f"{position.y:>12.3f}"
                f"{position.z:>12.3f}\n"
            )

        # ---------------------------------
        # Testergebnisse
        # ---------------------------------

        file.write("\n\n")
        file.write("Testergebnisse\n")
        file.write("--------------\n\n")
        file.write(results_text)


# ---------------- Phantom und Analyse ----------------

bbs = [
    BBConfig(
        name="Iso",
        offset_left_mm=0,
        offset_up_mm=0,
        offset_in_mm=0,
        bb_size_mm=5,
        rad_size_mm=14,
    ),
    BBConfig(
        name="1",
        offset_left_mm=20,
        offset_up_mm=0,
        offset_in_mm=40,
        bb_size_mm=5,
        rad_size_mm=14,
    ),
    BBConfig(
        name="2",
        offset_left_mm=0,
        offset_up_mm=-30,
        offset_in_mm=60,
        bb_size_mm=5,
        rad_size_mm=14,
    ),
    BBConfig(
        name="3",
        offset_left_mm=-20,
        offset_up_mm=0,
        offset_in_mm=-30,
        bb_size_mm=5,
        rad_size_mm=14,
    ),
    BBConfig(
        name="4",
        offset_left_mm=0,
        offset_up_mm=30,
        offset_in_mm=-50,
        bb_size_mm=5,
        rad_size_mm=14,
    ),
]

wl = WinstonLutzMultiTargetMultiField(
    str(DICOM_DIR)
)

wl.analyze(
    bb_arrangement=bbs,
    bb_proximity_mm=BB_PROXIMITY_MM,
    is_low_density=False,
    is_open_field=False,
)

write_results(
    wl,
    TXT_FILE,
)

write_custom_pdf(
    wl,
    str(PDF_FILE),
)
