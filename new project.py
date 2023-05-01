import cv2
import pytesseract
import csv
import datetime
import os.path

# Initialize Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Load  haar cascade classifier for car detection
car_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

# Define the set of number plates to detect
number_plates = ['AA 000 AA', 'HR26DK83##', 'DL4CAF4943', 'KA 09 G 8888']

# Check if CSV file already exists
file_exists = os.path.isfile('number_plates.csv')

# Open CSV file for writing or appending
with open('number_plates.csv', mode='a' if file_exists else 'w') as csv_file:
    # Define CSV fieldnames
    fieldnames = ['Category', 'Number', 'Date', 'Time']

    # Create CSV writer object
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write CSV header row if file is newly created
    if not file_exists:
        writer.writeheader()

    # Open video capture device
    cap = cv2.VideoCapture(0)

    # Define path for saving images
    save_path = 'images/'

    while True:
        # Read a frame from video capture device
        ret, frame = cap.read()

        # Convert frame to grayscale for car detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in frame using pre-trained cascade classifier
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        # Iterate over detected cars
        for (x, y, w, h) in cars:
            # Crop car image region from the frame
            car_image = frame[y:y+h, x:x+w]

            # Convert car image to grayscale for number plate detection
            car_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to car image for better number plate detection
            _, thresh = cv2.threshold(car_gray, 150, 255, cv2.THRESH_BINARY)

            # Apply morphological transformations to car image for better number plate detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.erode(thresh, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            # Detect contours in car image
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Display car image with number plate bounding box
            cv2.imshow('car', car_image)

            # Iterate over contours in car image
            for contour in contours:
                # Get bounding rectangle of contour
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)

                # Check if contour is likely to be number plate (based on aspect ratio and size)
                if w_c > 80 and h_c > 10 and w_c < 400 and h_c < 100 and w_c / h_c > 2:
                    # Crop number plate image region from the car image
                    number_plate_image = car_image[y_c:y_c+h_c, x_c:x_c+w_c]

                    # Apply OCR to number plate image to extract text
                    number_plate_text = pytesseract.image_to_string(number_plate_image, lang='eng', config='--psm 11')

                    # Check if detected number plate is in the set of number plates to detect
                    if number_plate_text.strip() in number_plates:
                        print(number_plate_text.strip())
                        # Get current timestamp
                        date = datetime.datetime.now().strftime("%Y-%m-%d")
                        time = datetime.datetime.now().strftime("%H:%M:%S")

                        # Write number plate text and timestamp to CSV file
                        writer.writerow({'Category': 'clg bus','Number': number_plate_text.strip(), 'Date': date,'Time': time})

                        
        # Wait for user input to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                    
