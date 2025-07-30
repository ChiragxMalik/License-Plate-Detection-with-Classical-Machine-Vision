import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract

# Set path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup folders
input_folder = 'Input'
output_folder = 'Output'
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(input_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f"\nWorking on: {image_file}")
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Can't read this image: {image_file}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # Reduce noise
        filtered = cv2.bilateralFilter(equalized, 11, 17, 17)

        # Find edges
        edges = cv2.Canny(filtered, 30, 200)

        # Find shapes in the image
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

        # Look for rectangle shapes that could be license plates
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        plate_contour = None

        for contour in top_contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
            
            # Check if it's a 4 sided shape with right proportions
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w/h
                area = cv2.contourArea(contour)
                if 1.5 <= aspect_ratio <= 5.0 and area > 1000:
                    plate_contour = approx
                    break

        if plate_contour is None:
            print("No license plate found")
            warped = np.zeros_like(image)
            cropped_image = warped
            plate_text = "No text found"
        else:
            print("Found a license plate!")
            
            # Fix perspective and straighten the plate
            pts = plate_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # Sort corners: top left, top right, bottom right, bottom left
            sum_pts = pts.sum(axis=1)
            rect[0] = pts[np.argmin(sum_pts)]
            rect[2] = pts[np.argmax(sum_pts)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Calculate size of the plate
            tl, tr, br, bl = rect
            width = max(int(np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)),
                        int(np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)))
            height = max(int(np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)),
                         int(np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)))

            # Transform the plate to straight view
            dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
            transform = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, transform, (width, height))

            # Crop the plate
            crop_top = int(0.15 * height)
            crop_bottom = int(0.30 * height)
            cropped_image = warped[crop_top:height - crop_bottom, :]

            # Read the text from the plate
            try:
                gray_plate = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate_text = pytesseract.image_to_string(thresh, config='--psm 8').strip()
                print(f"Plate says: {plate_text}")
            except:
                print("Couldn't read the text")
                plate_text = "Read error"

        # Add info text to images
        info_image = warped.copy()
        cv2.putText(info_image, f'Size: {width}x{height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        text_image = cropped_image.copy()
        cv2.putText(text_image, f'Text: {plate_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save all versions
        base_name = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(output_folder, f'{base_name}_extracted.png'), warped)
        cv2.imwrite(os.path.join(output_folder, f'{base_name}_cropped.png'), cropped_image)
        cv2.imwrite(os.path.join(output_folder, f'{base_name}_with_size.png'), info_image)
        cv2.imwrite(os.path.join(output_folder, f'{base_name}_with_text.png'), text_image)

        # Show processing steps 
        if True:  # Change to False if you dont want to see the steps
            steps = [image, gray, equalized, filtered, edges, contour_img, warped, cropped_image, info_image, text_image]
            step_names = ["Original", "Grayscale", "Better contrast", "Less noise", "Edges", 
                         "Shapes found", "Plate extracted", "Cropped plate", "With size", "With text"]
            
            plt.figure(figsize=(15, 15))
            for i in range(len(steps)):
                plt.subplot(4, 3, i+1)
                if len(steps[i].shape) == 2:
                    plt.imshow(steps[i], cmap='gray')
                else:
                    plt.imshow(cv2.cvtColor(steps[i], cv2.COLOR_BGR2RGB))
                plt.title(step_names[i])
                plt.axis('off')
            plt.tight_layout()
            plt.show()