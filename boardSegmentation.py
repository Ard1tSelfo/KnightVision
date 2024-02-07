import itertools

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np


def segment_chessboard(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, height=500)
    image_height = image.shape[0]
    image_width = image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ---------------------------
    plt.figure()
    plt.imshow(image)
    plt.show()
    plt.imshow(gray)
    plt.show()
    # ---------------------------

    # Parameters to adjust
    blur_kernels = [(5, 5), (7, 7), (9, 9)]  # Example GaussianBlur kernel sizes
    canny_thresholds = [(50, 150), (100, 170), (150, 250), (100, 200)]  # Example Canny thresholds
    approx_accuracies = [0.01, 0.02, 0.04, 0.06]  # Example approximation accuracies

    #gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #edged = cv2.Canny(gray, 100, 170)

    screen_cnt = None  # Initialize the screen contour variable

    # Attempt to adjust parameters dynamically
    for blur_kernel in blur_kernels:
        if screen_cnt is not None: break
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, blur_kernel, 0)

        for canny_threshold in canny_thresholds:
            if screen_cnt is not None: break
            edged = cv2.Canny(gray, canny_threshold[0], canny_threshold[1])

            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

            for approx_accuracy in approx_accuracies:
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, approx_accuracy * peri, True)

                    if len(approx) == 4:
                        screen_cnt = approx
                        break  # Exit the loop if a valid contour is found

                if screen_cnt is not None: break  # Exit the approximation accuracy loop if a valid contour is found

    # Handle case where no suitable contour was found after adjustments
    if screen_cnt is None:
        print("No suitable chessboard contour found after adjustments.")
        return None

    # ---------------------------
    plt.imshow(gray)
    plt.show()
    plt.imshow(edged)
    plt.show()
    # ---------------------------

    pts1 = np.float32(screen_cnt.reshape(4, 2))
    pts1 = pts1[np.lexsort((pts1[:, 1], pts1[:, 0]))]
    pts2 = []
    # find the appropriate orientation of the reference transformation coordinate system
    for corner in pts1:
        x = image_width - corner[0]
        y = image_height - corner[1]
        if x > corner[0] and y > corner[1]:
            pts2.append([0, 0])
        if x > corner[0] and y < corner[1]:
            pts2.append([0, 500])
        if x < corner[0] and y > corner[1]:
            pts2.append([500, 0])
        if x < corner[0] and y < corner[1]:
            pts2.append([500, 500])

    pts2 = np.float32(pts2)

    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, m, (500, 500))

    plt.imshow(dst)
    plt.axis("off")
    plt.show()

    return dst

# STILL PROTOTYPE
def segment_chessboard_hough(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150, apertureSize=3)

    # ---------------------------
    plt.figure()
    plt.imshow(image)
    plt.show()
    plt.imshow(gray)
    plt.show()
    plt.imshow(edged)
    plt.show()
    # ---------------------------
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 150)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Lines')
        plt.show()
    else:
        print("No lines detected")
        return None

    intersections = []
    for line1, line2 in itertools.combinations(lines, 2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        # Solve the system of equations A*x = b
        intersection = np.linalg.solve(A, b)
        if np.all(np.isfinite(intersection)):
            intersections.append(intersection)

    if not intersections:
        print("No intersections found")
        return None

    intersections = np.array(intersections).reshape(-1, 2)
    top_left = np.min(intersections, axis=0)
    bottom_right = np.max(intersections, axis=0)
    # Draw the detected corners for visualization
    for point in [top_left, bottom_right]:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    cv2.imshow('Detected Corners', image)

    # - Find intersections of the lines to get points.
    # - Determine the outer corners of the chessboard from these points.
    # - Apply perspective transformation based on the identified corners.

    return image  # Placeholder for the transformed chessboard image

