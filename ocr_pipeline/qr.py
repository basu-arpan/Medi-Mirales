"""
QR code generation for contactless prescription sharing — built entirely on
OpenCV's QRCodeEncoder, so the project stays on the OpenCV/Tesseract stack
without an extra third-party QR dependency.

Each processed prescription gets a QR code that encodes a URL back to
/record/<record_id> — scanning it with any phone camera opens a clean,
read-only view of the structured prescription. This is deliberately a URL
(not the raw prescription data) so that:
  - the QR stays small/scannable even for prescriptions with many medicines
  - the record can still be updated by staff after generation (e.g. a
    correction) without needing to reprint the QR
  - there's no PHI sitting directly inside a QR image that could be copied
    or stored separately from the system of record
"""

import os

import cv2
import numpy as np

# Base URL the QR encodes. In production this would be the deployed origin;
# for local pilot testing it points at localhost.
BASE_URL = os.environ.get("MEDI_MIRACLES_BASE_URL", "http://localhost:5000")

MODULE_SCALE = 10   # pixels per QR "module" (the smallest black/white square)
BORDER_MODULES = 4  # quiet-zone border, in modules, required for reliable scans

# Brand colors for the QR -- kept high-contrast since low contrast is the
# single biggest cause of failed scans on a clinic-staff phone camera.
INK = (58, 33, 20)       # BGR for --ink (#14213A)
PAPER_WHITE = (255, 255, 255)


def _render_qr_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Take OpenCV's raw 0/255 module grid and render it as a scaled, bordered,
    brand-colored PNG-ready BGR image.
    """
    h, w = matrix.shape
    scaled = cv2.resize(
        matrix, (w * MODULE_SCALE, h * MODULE_SCALE), interpolation=cv2.INTER_NEAREST
    )

    border_px = BORDER_MODULES * MODULE_SCALE
    canvas_h, canvas_w = scaled.shape[0] + 2 * border_px, scaled.shape[1] + 2 * border_px
    canvas = np.full((canvas_h, canvas_w, 3), PAPER_WHITE, dtype=np.uint8)

    # In the raw matrix, 0 = black module (data), 255 = white module
    colored_modules = np.where(scaled[..., None] == 0, INK, PAPER_WHITE).astype(np.uint8)
    canvas[border_px:border_px + scaled.shape[0], border_px:border_px + scaled.shape[1]] = colored_modules

    return canvas


def generate_qr_for_record(record_id: str, structured: dict, output_dir: str) -> str:
    """
    Generate and save a QR code image for a given record, returning the
    static-relative path the frontend can render directly into an <img> tag.
    """
    target_url = f"{BASE_URL}/record/{record_id}"

    params = cv2.QRCodeEncoder_Params()
    params.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_M
    params.mode = cv2.QRCodeEncoder_MODE_BYTE

    # Explicitly size the QR version rather than relying on auto-detection
    # (version = -1). Several OpenCV releases (4.9-4.11 at time of writing)
    # have a known bug where auto-detected version silently produces an
    # unscannable code once the payload exceeds ~135 characters — no error
    # is raised, the image just doesn't decode on a real phone. Our payload
    # is a short URL, but picking the version explicitly from the actual
    # string length costs nothing and removes the failure mode entirely.
    # Version N supports up to (roughly) 16*N + 20 alphanumeric/byte chars
    # at correction level M — we pad generously and cap at QR's max (40).
    estimated_version = max(1, min(40, (len(target_url) // 12) + 2))
    params.version = estimated_version

    encoder = cv2.QRCodeEncoder_create(params)
    matrix = encoder.encode(target_url)

    qr_image = _render_qr_matrix(matrix)

    fname = f"{record_id}_qr.png"
    path = os.path.join(output_dir, fname)
    cv2.imwrite(path, qr_image)

    return f"/static/qrcodes/{fname}"
