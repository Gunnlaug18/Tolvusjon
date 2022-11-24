import cv2
import time
import numpy as np

cap = cv2.VideoCapture(1)

def brightest_spot(gray,frame):
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return cv2.circle(frame, maxLoc, 5, (0, 255, 0), 2)

def reddest_spot(frame,the_circle,red_mask,hsv):
    (_, _, _, point) = cv2.minMaxLoc(hsv[:, :, 1], red_mask)
    the_circle = np.copy(the_circle)
    the_circle = cv2.circle(frame, point, 8, (0, 255, 0), 2)
    return the_circle


def main():

    while(True):
        # Start time
        start = time.time()
        ret, frame = cap.read()
        # End time
        end = time.time()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #like hello world assignment but in computer visionif we have color then we look for the redest spot and if we have gray then we look for the lightest spot 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # MASK in the red range
        lower = np.array([0,100,100])
        upper = np.array([0,255,255])
        red_mask = cv2.inRange(hsv, lower, upper)
        the_fig = brightest_spot(gray,frame)
        the_fig = reddest_spot(frame,the_fig,red_mask,hsv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        framespersecond = int(1/(end-start))
        cv2.putText(the_fig, "FPS:" + str(framespersecond), (10,450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame',the_fig)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows

main()