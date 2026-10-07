# Project 4 — Image / Text Recognition

## DecodeLabs Artificial Intelligence Internship

This project implements a basic Optical Character Recognition (OCR) pipeline using Python, OpenCV and Tesseract.

## Pipeline
Input Image → Grayscale → Gaussian Blur → Adaptive Thresholding → Deskewing → Tesseract OCR → Confidence → Extracted Text

## Technologies
- Python
- OpenCV
- NumPy
- Pytesseract
- Tesseract OCR

## Structure
```text
Project-4/
├── main.py
├── requirements.txt
├── README.md
├── sample_images/
│   └── sample_text.png
└── output/
```

## Installation
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tesseract OCR must also be installed separately.

## Run
```powershell
python main.py "sample_images\sample_text.png"
```

If needed:
```powershell
python main.py "sample_images\sample_text.png" --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Validation
The application uses an 80% OCR confidence threshold:
- 80% or higher: PASS
- Below 80%: REVIEW

## Internship
**Artificial Intelligence Internship — DecodeLabs**

**Project 4: Image / Text Recognition**

## Author
**Pihu Verma**

GitHub: https://github.com/Pihu-v17/Decode-Labs-AI-Internship
