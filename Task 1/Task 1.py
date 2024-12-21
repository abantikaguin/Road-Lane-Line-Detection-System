import cv2
import numpy as np

# Define the region of interest for lane detection
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Edge detection using Canny algorithm
def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Perform Hough Transform to detect lane lines
def hough_transform(img, rho=1, theta=np.pi/180, threshold=50, min_line_len=100, max_line_gap=160):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Draw the lane lines on the image
def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return img

# Main lane detection pipeline
def lane_detection_pipeline(image):
    # 1. Detect edges
    edges = detect_edges(image)

    # 2. Define region of interest (ROI)
    height, width = edges.shape
    vertices = np.array([[(100, height), (width - 100, height), (width // 2, height // 2)]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)

    # 3. Apply Hough Transform to detect lines
    lines = hough_transform(roi_edges)

    # 4. Draw the detected lines on the original image
    lane_image = draw_lines(np.copy(image), lines)

    return lane_image

# Test the system on a single image
def test_lane_detection(image_path):
    image = cv2.imread(image_path)
    lane_image = lane_detection_pipeline(image)

    # Show the result
    cv2.imshow("Lane Detection", lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the system on a video stream (optional)
def test_video_lane_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        lane_image = lane_detection_pipeline(frame)
        
        # Show the result for each frame
        cv2.imshow("Lane Detection", lane_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()



video_path = '42483-431756068_tiny.mp4'  # Replace with your video path
test_video_lane_detection(video_path)


