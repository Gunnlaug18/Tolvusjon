import cv2
import time
import numpy as np

cap = cv2.VideoCapture(1)

def brightest_spot_pixle(frame,gray):
    position = (0, 0)
    brightest = 0
    (x, y) = gray.shape
    for i in range(x):
        for j in range(y):
            if (gray[i, j] > brightest):
                brightest = gray[i, j]
                position = (j, i)
    gray = cv2.circle(frame, position, 8, (0, 255, 0), 2)
    return gray

def reddest_spot(the_fig,hsv):
    position = (0, 0)
    reddest = [0,100,100]
    (x,y,z) = hsv.shape
    for i in range(x):
        for j in range(y):
            if (hsv[i, j][2] >= reddest[2] and hsv[i,j][1] >= reddest[1] and hsv[i,j][0] == 0):
                reddest = hsv[i, j]
                position = (j, i)
    the_fig = cv2.circle(the_fig, position, 8, (0, 0, 255), 2)
    return the_fig

def main():

    while(True):
        # Start time
        start = time.time()
        ret, frame = cap.read()
        # End time
        # framespersecond= int(cap.get(cv2.CAP_PROP_FPS))

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #like hello world assignment but in computer visionif we have color then we look for the redest spot and if we have gray then we look for the lightest spot 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        the_fig = brightest_spot_pixle(frame,gray)
        the_fig = reddest_spot(the_fig,hsv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        end = time.time()
        framespersecond = int(1/(end-start))
        cv2.putText(the_fig, "FPS:" + str(framespersecond), (10,450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame',the_fig)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows

main()