import cv2
import time
import numpy as np
from collections import defaultdict
#https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv?fbclid=IwAR2Y4vcaDW7KHC277AQC2F5Oux_63F7_Wv8HcWRx_J9AcQZ_mbd5vRVYLwY
cap = cv2.VideoCapture(1)

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def draw_lines(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """  
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, 1)

def main():
    while(True):
        start = time.time()
        _, frame = cap.read()
        # frame = cv2.imread("argsmall.png")
        frame = frame[100:400,100:600]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #canny
        canny = cv2.Canny(gray,150,250,apertureSize=3)

        # img = im.copy()
        lines = cv2.HoughLines(canny,1,np.pi/90,100)
        if lines is not None and len(lines) > 2:
            segmented = segment_by_angle_kmeans(lines)
            intersections = segmented_intersections(segmented)
            # vertical_lines = segmented[1]
            draw_lines(frame, lines)
            # for point in intersections:
            if len(intersections) == 4: 
                for x in intersections:
                    cv2.circle(frame, x[0], 1, (0, 255, 0), 2)
    
                intersections = np.array(([x[0] for x in intersections]))
                
                print(intersections)
                new_corners =  np.float32([[0,0],[639,0],[0, 479],[639,479]])

                h, status = cv2.findHomography(intersections, new_corners)

                im_out = cv2.warpPerspective(frame, h, (640,480))
                cv2.imshow("out_img",im_out)
        frame = cv2.resize(frame, (640, 480))
        font = cv2.FONT_HERSHEY_SIMPLEX

        frame = cv2.resize(frame, (640, 480))
        end = time.time()
        framespersecond = int(1/(end-start))
        cv2.putText(frame, "FPS:" + str(framespersecond), (10,450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('houghlines',frame)
        print(end-start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows

main()