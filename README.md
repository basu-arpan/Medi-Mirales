# Medi-Miracles — ML-Based Medical Prescription Digitization System

Turns a photo of a handwritten prescription into a structured digital
record — patient details, medicines with dosage/frequency/duration — plus a
QR code for contactless sharing with the patient or pharmacy.

Built with **Python · Flask · OpenCV · Tesseract OCR (pytesseract)**.

---

## How it works

```
 Upload          OpenCV preprocessing          Tesseract OCR        Structuring        QR code
┌────────┐      ┌──────────────────────┐      ┌───────────┐      ┌──────────────┐    ┌─────────┐
│ scan / │ ───▶ │ grayscale → denoise  │ ───▶ │ image_to_ │ ───▶ │ regex-based  │ ──▶│ encodes │
│ photo  │      │ → deskew → contrast  │      │ string /  │      │ field +      │    │ /record/│
└────────┘      │ → binarize → cleanup │      │ image_to_ │      │ medicine     │    │ <id>    │
                └──────────────────────┘      │ data      │      │ extraction   │    └─────────┘
                                               └───────────┘      └──────────────┘
```

1. **Preprocessing** (`ocr_pipeline/preprocess.py`) — a 6-stage OpenCV
   pipeline (grayscale, denoise, deskew, CLAHE contrast normalization,
   adaptive binarization, morphological cleanup) that cleans up
   handwritten/photographed prescriptions before they hit Tesseract. Every
   stage is saved to disk so the UI can show the image at each step.
2. **OCR extraction** (`ocr_pipeline/extract.py`) — runs Tesseract via
   `pytesseract`, using `image_to_data` to get per-word confidence scores
   (not a fabricated number), then parses the raw text into structured
   fields (patient name, age/sex, date, doctor, and a list of medicines
   with dosage/frequency/duration) using a defensive set of regexes that
   degrade gracefully — anything it can't confidently extract is left
   blank for staff to fill in, rather than guessing.
3. **QR generation** (`ocr_pipeline/qr.py`) — built on OpenCV's own
   `cv2.QRCodeEncoder` (no extra dependency needed). Encodes a URL to
   `/record/<id>`, not the raw data, so the QR stays small and the record
   can be corrected later without re-printing it.
4. **Flask app** (`app.py`) — ties it together: `/api/process` runs the
   full pipeline on an upload, `/record/<id>` is the human-friendly page a
   phone lands on after scanning the QR, `/api/records` powers the pilot
   activity dashboard.

## Project structure

```
medi-miracles/
├── requirements.txt
└── app/
    ├── app.py                  # Flask routes
    ├── ocr_pipeline/
    │   ├── preprocess.py       # OpenCV cleanup pipeline
    │   ├── extract.py          # Tesseract OCR + structuring
    │   └── qr.py               # QR code generation
    ├── templates/
    │   ├── index.html          # Upload + pilot dashboard
    │   ├── record.html         # QR-landing record view
    │   └── record_not_found.html
    ├── static/
    │   ├── css/style.css
    │   ├── js/app.js           # Upload UX, pipeline animation, rendering
    │   ├── uploads/            # original scans (gitignored contents)
    │   ├── processed/          # per-stage preprocessing previews
    │   ├── qrcodes/            # generated QR images
    │   └── records/            # one JSON file per processed prescription
    └── sample_data/
        └── test_prescription.png   # synthetic sample for testing
```

## Setup

Requires the Tesseract binary installed on the system (not just the Python
wrapper):

```bash
# Debian/Ubuntu
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

Then:

```bash
cd medi-miracles
pip install -r requirements.txt
cd app
python app.py
```

Visit `http://localhost:5000`.

## Notes on the included sample

`app/sample_data/test_prescription.png` is a synthetically generated test
image (printed text + artificial rotation/noise), used to validate the
pipeline end-to-end without needing a real patient prescription. Drop your
own scanned/photographed prescriptions into the upload zone to see it work
on real handwriting — accuracy depends heavily on handwriting legibility
and photo quality, which is exactly why the preprocessing stage exists.

## Extending it

- **Medicine vocabulary**: `extract.py` uses a small, high-precision pattern
  set for frequency/duration rather than a drug-name dictionary, since
  false positives are worse than a missed match when staff review every
  record anyway. Swap in a real drug database for higher recall.
- **Storage**: records are flat JSON files for pilot simplicity. For a
  real deployment, swap `RECORDS_DIR` for a proper database.
- **Auth & PHI**: there's no authentication layer here — this is a pilot
  scaffold, not a HIPAA-compliant system. Add access control before
  handling real patient data.
