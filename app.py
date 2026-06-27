"""
Medi-Miracles — ML-Based Medical Prescription Digitization System
Flask backend: handles uploads, runs the OCR pipeline, generates QR codes,
and serves the structured prescription record.
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from ocr_pipeline.preprocess import preprocess_image
from ocr_pipeline.extract import extract_text, structure_prescription
from ocr_pipeline.qr import generate_qr_for_record

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")
QR_DIR = os.path.join(BASE_DIR, "static", "qrcodes")
RECORDS_DIR = os.path.join(BASE_DIR, "static", "records")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

for d in (UPLOAD_DIR, PROCESSED_DIR, QR_DIR, RECORDS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB cap


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def record_path(record_id: str) -> str:
    return os.path.join(RECORDS_DIR, f"{record_id}.json")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_prescription():
    """
    Full pipeline endpoint:
    1. Save uploaded scan
    2. Run OpenCV preprocessing (deskew, denoise, binarize)
    3. Run Tesseract OCR on the cleaned image
    4. Parse raw text into a structured prescription record
    5. Generate a QR code pointing to the record
    6. Persist record as JSON, return everything to the frontend
    """
    start_time = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, BMP, TIFF, or WEBP."}), 400

    record_id = uuid.uuid4().hex[:10]
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{record_id}_original.{ext}"
    original_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(original_path)

    try:
        # Step 1 — preprocessing pipeline, returns stage-by-stage images + timing
        preprocess_result = preprocess_image(
            original_path, output_dir=PROCESSED_DIR, record_id=record_id
        )

        # Step 2 — OCR extraction on the cleaned image
        ocr_result = extract_text(preprocess_result["final_path"])

        # Step 3 — structure the raw OCR text into prescription fields
        structured = structure_prescription(ocr_result["raw_text"])

        # Step 4 — QR code for contactless sharing
        qr_rel_path = generate_qr_for_record(
            record_id=record_id,
            structured=structured,
            output_dir=QR_DIR,
        )

        elapsed = round(time.time() - start_time, 2)

        record = {
            "record_id": record_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processing_seconds": elapsed,
            "original_image": f"/static/uploads/{safe_name}",
            "pipeline_stages": preprocess_result["stage_previews"],
            "stage_timings_ms": preprocess_result["timings_ms"],
            "final_processed_image": preprocess_result["final_preview"],
            "ocr": {
                "raw_text": ocr_result["raw_text"],
                "confidence": ocr_result["confidence"],
                "word_count": ocr_result["word_count"],
            },
            "structured": structured,
            "qr_code": qr_rel_path,
        }

        with open(record_path(record_id), "w") as f:
            json.dump(record, f, indent=2)

        return jsonify(record), 200

    except Exception as exc:
        app.logger.exception("Processing failed")
        return jsonify({"error": f"Processing failed: {str(exc)}"}), 500


@app.route("/api/record/<record_id>", methods=["GET"])
def get_record(record_id):
    """Fetch a previously processed record — this is what the QR code resolves to."""
    path = record_path(record_id)
    if not os.path.exists(path):
        abort(404)
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/record/<record_id>")
def view_record(record_id):
    """Human-friendly page a phone camera lands on after scanning the QR code."""
    path = record_path(record_id)
    if not os.path.exists(path):
        return render_template("record_not_found.html", record_id=record_id), 404
    with open(path) as f:
        record = json.load(f)
    return render_template("record.html", record=record)


@app.route("/api/records", methods=["GET"])
def list_records():
    """Pilot dashboard data — every record processed so far, most recent first."""
    records = []
    for fname in sorted(os.listdir(RECORDS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(RECORDS_DIR, fname)) as f:
                data = json.load(f)
                records.append({
                    "record_id": data["record_id"],
                    "created_at": data["created_at"],
                    "processing_seconds": data["processing_seconds"],
                    "patient_name": data["structured"].get("patient_name") or "—",
                    "confidence": data["ocr"]["confidence"],
                })
    return jsonify(records)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
