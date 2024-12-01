import cv2
import numpy as np

def region_of_interest(image):
    # Define a mask for the region of the image where we want to detect lane lines
    height, width = image.shape[:2]
    polygons = np.array([
        [(0, height), (width, height), (width // 2, height // 2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)
    
    return edges

def draw_lines(image, lines):
    # Draw lines on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

def detect_lanes(image):
    # Process the image to detect lanes
    edges = process_image(image)
    # Define the region of interest
    roi = region_of_interest(edges)
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    # Draw detected lines on original image
    draw_lines(image, lines)
    return image

# Main function to run the lane detection on a video
def main():
    # Correct the file path; remove extra space
    cap = cv2.VideoCapture(r"C:\Users\Abantika\Downloads\42483-431756068_tiny.mp4")  # Use raw string to avoid escape characters

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run the lane detection on the frame
        lane_image = detect_lanes(frame)

        # Display the result
        cv2.imshow('Lane Detection', lane_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
