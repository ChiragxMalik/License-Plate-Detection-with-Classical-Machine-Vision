# License-Plate-Detection-with-Classical-Machine-Vision


A project a did for my Machine Vision course that detects and reads license plates using traditional image processing techniques. no deep learning, Just pure OpenCV, computer vision fundamentals, and OCR.

While I'm learning and using modern tools, this project helped me understand the foundational concepts behind how machines actually "see" images.

## How It Works

The system processes images through these steps:

1. **Image Preparation**  
   - Convert to grayscale
   - Enhance contrast using CLAHE
   - Reduce noise with bilateral filtering

2. **Plate Detection**  
   - Find edges using Canny detector
   - Identify contours in the image
   - Filter for rectangular shapes (4-sided polygons)
   - Select the most likely license plate candidate

3. **Plate Processing**  
   - Fix perspective distortion
   - Crop to the plate region
   - Apply thresholding for better OCR

4. **Text Recognition**  
   - Extract text using Tesseract OCR
   - Display and save results


## Output

<img width="1920" height="1080" alt="Screenshot (204)" src="https://github.com/user-attachments/assets/aa4009cb-a599-4467-9a4c-d8c9a537c609" />

<img width="1920" height="1080" alt="Screenshot (203)" src="https://github.com/user-attachments/assets/6ca15c45-d140-4406-a1ac-a8db2f262245" />


## Installation & Usage

### Requirements
- Python 3.6+
- Tesseract OCR

### Setup
1. Install Tesseract:  
   [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
