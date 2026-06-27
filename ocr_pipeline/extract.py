"""
OCR extraction (Tesseract via pytesseract) and structuring of raw prescription
text into clean fields: patient name, date, medicines (with dosage/frequency/
duration), and doctor notes.

Handwriting OCR is inherently noisy, so `structure_prescription` is written
defensively: every regex has a fallback, and we never raise on a field we
can't find — we just leave it blank for clinic staff to fill in manually.
That graceful-degradation behavior is what makes the 45-second number
trustworthy: the system tells you what it's confident about instead of
guessing silently.
"""

import re

import cv2
import pytesseract
from pytesseract import Output


# Common drug-form and frequency vocabulary used to anchor medicine-line
# parsing. This is intentionally a small, high-precision list rather than a
# huge dictionary — false positives (treating a stray word as a drug name)
# are worse than missing an occasional uncommon drug, since staff review
# every record before it's finalized anyway.
DOSAGE_UNIT_PATTERN = r"(?:mg|mcg|ml|g|IU|units?)"
FREQUENCY_PATTERNS = [
    (r"\bOD\b|\bonce\s+a?\s*day\b", "Once daily (OD)"),
    (r"\bBD\b|\bBID\b|\btwice\s+a?\s*day\b", "Twice daily (BD)"),
    (r"\bTDS\b|\bTID\b|\bthrice\s+a?\s*day\b|\b3\s*times?\s+a?\s*day\b", "Three times daily (TDS)"),
    (r"\bQID\b|\b4\s*times?\s+a?\s*day\b", "Four times daily (QID)"),
    (r"\bHS\b|\bat\s+night\b|\bbedtime\b", "At bedtime (HS)"),
    (r"\bSOS\b|\bas\s+needed\b|\bPRN\b", "As needed (SOS/PRN)"),
]
DURATION_PATTERN = r"(\d+)\s*(days?|weeks?|months?)"


def extract_text(image_path: str) -> dict:
    """
    Run Tesseract on the preprocessed image and return both the raw text
    and a confidence score derived from Tesseract's own per-word confidences.

    We use image_to_data (not just image_to_string) so we get per-word
    confidence — averaging that gives a meaningful "how sure are we" number
    to surface in the UI, rather than a fabricated metric.
    """
    img = cv2.imread(image_path)

    # PSM 6 = "assume a single uniform block of text", which suits a
    # prescription pad better than the default PSM 3 page-segmentation mode,
    # which can over-split short handwritten lines into separate blocks.
    custom_config = r"--oem 3 --psm 6"

    raw_text = pytesseract.image_to_string(img, config=custom_config)

    data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    words = [w for w in raw_text.split() if w.strip()]

    return {
        "raw_text": raw_text.strip(),
        "confidence": avg_confidence,
        "word_count": len(words),
    }


def _extract_patient_name(text: str) -> str:
    match = re.search(r"(?:patient|name|pt\.?)\s*[:\-]?\s*([A-Za-z][A-Za-z.\s]{1,40}?)(?=\n|,|\s{2,}|$|\d)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".").title()
    return ""


def _extract_date(text: str) -> str:
    match = re.search(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b", text)
    return match.group(1) if match else ""


def _extract_doctor(text: str) -> str:
    match = re.search(r"(?:dr\.?|doctor)\s*[:\-]?\s*([A-Za-z][A-Za-z.\s]{1,40}?)(?=\n|,|\s{2,}|$)", text, re.IGNORECASE)
    if match:
        name = match.group(1).strip().rstrip(".")
        if name:
            return ("Dr. " + name).title().replace("Dr. Dr.", "Dr.")
    return ""


def _extract_age_sex(text: str) -> str:
    match = re.search(r"\b(\d{1,3})\s*(?:y|yrs?|years?)?\s*[\/,]\s*(M|F|Male|Female)\b", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} / {match.group(2).upper()[0]}"
    return ""


def _parse_medicine_lines(text: str) -> list:
    """
    Heuristically split the OCR text into individual medicine entries.

    Strategy: scan line by line. A line is treated as the START of a new
    medicine entry only if it contains a dosage unit (mg/ml/mcg/etc) OR
    starts with a numbered marker AND has more than just the marker itself
    (so a bare "Rx" header doesn't spawn an empty entry). Any other
    non-header line is treated as a continuation of the current medicine —
    handwriting frequently wraps "BD for 7 days" onto its own line under
    the drug name, so continuations must NOT require a dosage unit to
    attach to the entry above them.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    medicines = []

    bullet_pattern = re.compile(r"^\s*(?:\d+[\.\)]|[-•*])\s*", re.IGNORECASE)
    rx_header_pattern = re.compile(r"^\s*Rx[:\.]?\s*$", re.IGNORECASE)
    metadata_pattern = re.compile(r"patient|date|doctor|dr\.|age|sex|hospital|clinic", re.IGNORECASE)
    dosage_signal = re.compile(r"\d+\s*" + DOSAGE_UNIT_PATTERN, re.IGNORECASE)

    current = None
    for line in lines:
        # Skip a bare "Rx" header line entirely — it's a section label,
        # not a medicine, and has nothing left once the bullet is stripped.
        if rx_header_pattern.match(line):
            continue

        # Skip metadata lines (patient/doctor/date/etc) unless they also
        # carry a dosage signal (rare, but a medicine line could mention
        # "as advised by doctor" — dosage signal takes priority).
        if metadata_pattern.search(line) and not dosage_signal.search(line):
            continue

        stripped_of_bullet = bullet_pattern.sub("", line).strip()
        has_bullet = bool(bullet_pattern.match(line))
        has_dosage = bool(dosage_signal.search(line))

        # A genuinely new medicine entry needs real content after the
        # bullet/number is removed, and either a bullet marker or a dosage
        # unit as evidence it's naming a drug rather than continuing one.
        is_new_entry = bool(stripped_of_bullet) and (has_bullet or has_dosage)

        if is_new_entry:
            if current:
                medicines.append(current)
            current = {"raw_line": stripped_of_bullet, "name": "", "dosage": "", "frequency": "", "duration": ""}

            dose_match = re.search(
                r"([A-Za-z][A-Za-z0-9\- ]{1,30}?)\s+(\d+\s*" + DOSAGE_UNIT_PATTERN + ")",
                stripped_of_bullet, re.IGNORECASE
            )
            if dose_match:
                current["name"] = dose_match.group(1).strip().title()
                current["dosage"] = dose_match.group(2).strip()
            else:
                current["name"] = stripped_of_bullet.title()

            for pattern, label in FREQUENCY_PATTERNS:
                if re.search(pattern, stripped_of_bullet, re.IGNORECASE):
                    current["frequency"] = label
                    break

            dur_match = re.search(DURATION_PATTERN, stripped_of_bullet, re.IGNORECASE)
            if dur_match:
                current["duration"] = f"{dur_match.group(1)} {dur_match.group(2)}"

        elif current:
            # Continuation line (e.g. "BD for 7 days" wrapped under the
            # drug name) — fill in whichever fields are still missing.
            current["raw_line"] += " " + line
            for pattern, label in FREQUENCY_PATTERNS:
                if not current["frequency"] and re.search(pattern, line, re.IGNORECASE):
                    current["frequency"] = label
                    break
            dur_match = re.search(DURATION_PATTERN, line, re.IGNORECASE)
            if not current["duration"] and dur_match:
                current["duration"] = f"{dur_match.group(1)} {dur_match.group(2)}"

    if current:
        medicines.append(current)

    return medicines


def structure_prescription(raw_text: str) -> dict:
    """
    Convert raw OCR text into a structured record ready for display,
    QR encoding, and pharmacy hand-off.
    """
    return {
        "patient_name": _extract_patient_name(raw_text),
        "age_sex": _extract_age_sex(raw_text),
        "date": _extract_date(raw_text),
        "doctor_name": _extract_doctor(raw_text),
        "medicines": _parse_medicine_lines(raw_text),
        "raw_text_fallback": raw_text,
    }
