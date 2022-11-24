import cv2
import time
import numpy as np
from numpy import empty, linalg as LA
from sklearn.linear_model import LinearRegression, RANSACRegressor
import math
import time
cap = cv2.VideoCapture(1)

def best_points(points):
    most_n_inliers = 0
    best_inliers = []
    for k in range(50):
        if len(points) == 0:
            break
        inliers = []
        N_inliers = 0
        rand_point_1 = np.random.randint(0, len(points))
        rand_point_2 = np.random.randint(0, len(points))
        p1 = points[rand_point_1]
        p2 = points[rand_point_2]
        inliers.append(p1)
        inliers.append(p2)
        for i in range(0,points.shape[0],30):
            if(len(points) != 0):
                point = points[i]
                #https://www.youtube.com/watch?v=tYUtWYGUqgw
                l_1_2 = np.linalg.norm(np.cross(p2-p1,p1-point))/np.linalg.norm(p2-p1)
                if (l_1_2 < 2):
                    N_inliers +=1
                    inliers.append(point)

        if(N_inliers > most_n_inliers):
            most_n_inliers = N_inliers
            best_inliers = inliers

    best_inliers = np.array(best_inliers)
    return best_inliers

def draw_lines(frame,canny,best_inliers):
    draw_x = np.arange(0, canny.shape[1], 1)
    if (max(best_inliers[:,1]) - min(best_inliers[:,1]) < 50):
        draw_y = np.polyval(np.polyfit(best_inliers[:,0],best_inliers[:,1],1),draw_x)
        draw_points = (np.asarray([draw_y, draw_x]).T).astype(np.int32) 
        cv2.polylines(frame, [draw_points], False, (255,0,0),2) 
    else:
        draw_y = np.polyval(np.polyfit(best_inliers[:,1],best_inliers[:,0],1),draw_x)
        draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32) 
        cv2.polylines(frame, [draw_points], False, (255,0,0),2) 
    

def main():

    while(True):
        start = time.time()
        _, frame = cap.read()
        # frame = cv2.imread("argsmall.png",1)
        # CANNY
        frame = frame[100:400,100:600]
        canny = cv2.Canny(frame,50,150,apertureSize=3,L2gradient=True)
        points = np.argwhere(canny != 0)
        best_inliers= best_points(points)
        draw_lines(frame,canny,best_inliers)
        font = cv2.FONT_HERSHEY_SIMPLEX

        frame = cv2.resize(frame, (640, 480))
        end = time.time()
        framespersecond = int(1/(end-start))
        cv2.putText(frame, "FPS:" + str(framespersecond), (10,450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("frame",frame)
        print(end-start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows

main()