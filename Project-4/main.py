from pathlib import Path
import argparse
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CONFIDENCE_THRESHOLD = 80.0

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return gray, blurred, thresholded

def deskew_image(image):
    inverted = cv2.bitwise_not(image)
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 20:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def extract_text(image):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        try:
            conf = float(conf)
        except ValueError:
            continue
        if text and conf >= 0:
            words.append(text)
            confidences.append(conf)
    text = " ".join(words)
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return text, confidence

def save_results(text, confidence, processed):
    OUTPUT_DIR.mkdir(exist_ok=True)
    text_file = OUTPUT_DIR / "extracted_text.txt"
    image_file = OUTPUT_DIR / "processed_image.png"
    text_file.write_text(
        "DECODELABS PROJECT 4 - OCR OUTPUT\n" +
        "=" * 50 + "\n\n" +
        f"Confidence: {confidence:.2f}%\n\n" +
        "Extracted Text:\n" + "-" * 50 + "\n" +
        text + "\n", encoding="utf-8"
    )
    cv2.imwrite(str(image_file), processed)
    return text_file, image_file

def main():
    parser = argparse.ArgumentParser(description="DecodeLabs Project 4 - OCR Recognition")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--tesseract", help="Optional path to Tesseract executable")
    args = parser.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    image_path = Path(args.image)
    print("=" * 65)
    print("        DECODELABS - PROJECT 4")
    print("        IMAGE / TEXT RECOGNITION")
    print("=" * 65)

    if not image_path.exists():
        print(f"\nERROR: Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print("\nERROR: Could not read the image.")
        return

    print("\n[1] Image loaded successfully.")
    print("[2] Converting image to grayscale...")
    _, _, thresholded = preprocess_image(image)
    print("[3] Applying Gaussian blur...")
    print("[4] Applying adaptive thresholding...")
    print("[5] Correcting image alignment...")
    processed = deskew_image(thresholded)
    print("[6] Running Tesseract OCR...")

    try:
        text, confidence = extract_text(processed)
    except pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract OCR was not found.")
        print('Use --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
        return

    print("\n" + "=" * 65)
    print("OCR RESULTS")
    print("=" * 65)
    print(f"\nConfidence Score: {confidence:.2f}%")
    print("Validation Status:", "PASS" if confidence >= CONFIDENCE_THRESHOLD else "REVIEW")
    print("\nExtracted Text:")
    print("-" * 65)
    print(text if text else "No readable text detected.")

    text_file, image_file = save_results(text, confidence, processed)
    print("\nOUTPUT FILES")
    print(f"Extracted text : {text_file}")
    print(f"Processed image: {image_file}")
    print("\nProject 4 OCR pipeline completed successfully.")

if __name__ == "__main__":
    main()
