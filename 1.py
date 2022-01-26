import cv2
from math import atan2, cos, sin, sqrt, pi
from cv2 import boundingRect
import numpy as np
 
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
    
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]
 
def getOrientation(countour, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(countour)
    data_countour = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_countour.shape[0]):
        data_countour[i,0] = countour[i,0,0]
        data_countour[i,1] = countour[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    # Phân tích thành phần chính (PCA) là một thủ tục thống kê trích xuất các tính năng quan trọng nhất của tập dữ liệu.
    # giảm dữ liệu 2D thành 1D
    # Thành phần chính thứ nhất: Trục chính của hình elip là hướng của phương sai cực đại và như chúng ta biết bây giờ,
    #         nó là hướng của thông tin cực đại, gọi là thành phần chính đầu tiên của dữ liệu.

    #Thành phần chính thứ hai: là hướng của phương sai cực đại vuông góc với hướng của thành phần chính thứ nhất

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_countour, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
    
    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    anpha = 0.02
    p1 = (cntr[0] + anpha * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + anpha * eigenvectors[0,1] * eigenvalues[0,0])
    #print(p1)
    #p2 = (cntr[0] - anpha * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - anpha * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    #drawAxis(img, cntr, p2, (0, 0, 255), 1)
    
    # atan2 (y, x) trả về góc θ giữa tia tới điểm (x, y) và trục x dương, giới hạn trong (−π, π]
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    angle = angle*180/3.14
    ## [visualization]
    
    # Label with the rotation angle
    #label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    #textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    #cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    
    return angle



def nothing(x):
    pass

#img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image',cv2.WINDOW_FREERATIO)

# create trackbars for color change+
cv2.createTrackbar('b1','image',176,255,nothing)
cv2.createTrackbar('b2','image',255,255,nothing)


# Load the image
img = cv2.imread("motor1.jpg")
#img = cv2.imread("download.png")
img = cv2.resize(img,(640,480))
 
# Was the image there?
# if img is None:
#   print("Error: File not found")
#   exit(0)
 
#cv.imshow('Input Image', img)
 
while True:
    img = cv2.imread("motor1.jpg")
    #img = cv2.imread("download.png")
    img = cv2.resize(img,(640,480))

    #xoay anh 90 do theo chieu kim dong ho
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.blur(gray,(5,5))

    #kernel = np.ones((5,5), np.uint8)

    #dilation = cv2.dilate(blur, kernel, iterations = 1)
    #erosion = cv2.erode(blur, kernel, iterations = 1)

    # xói mòn sau đó là sự giãn nở
    #opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # giãn nở theo sau là xói mòn
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    b1 = cv2.getTrackbarPos('b1','image')
    b2 = cv2.getTrackbarPos('b2','image')
    #print(str(b1))
    # Convert image to binary
    _, bw = cv2.threshold(gray, b1, b2,  cv2.THRESH_BINARY)
    #cv2.imshow('bw',bw)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    a=1
    for i, c in enumerate(contours):
            # Calculate the area of each contour
        area = cv2.contourArea(c)
        #if int(area) > 1000:
            #cv2.putText(img,str(int(area)),(20,a*30), 1,1,(0,0,255),1)
            #a+=1
        x,y,w,h = cv2.boundingRect(c)
        # Ignore contours that are too small or too large
        if area < 8000 or 10000 < area:
            continue

        # Draw each contour only for visualisation purposes
        #cv2.drawContours(img, c, -1, (0, 0, 255), 2)
        center = (x+w//2, y+h//2)
        #cv2.circle(img,center,1,(255,0,0),1)
        cv2.line(img,(x+w//2,y),(x+w//2,y+h),(0,0,0),2)
        #cv2.putText(img,str(int(w*h)),(400,a*30), 1,1,(0,0,255),1)
        #a+=1
        #object_mask= area
        #line_mask = (x+w//2,y),(x+w//2,y+h)
        #print(np.logical_and( object_mask, line_mask ))
        #cv2.bitwise_and(area,)
        #cv2.putText(img,str(area),((x,y)),1,2,(0,0,255),2)
        # Find the orientation of each shape
        #angle = getOrientation(c, img)
        #cv2.putText(img,str(int(angle)),((x,y)),1,2,(0,0,255),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #print(angle)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, b1, b2,  cv2.THRESH_BINARY)
    cv2.imshow('bw',bw)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    a=1
    for i, c in enumerate(contours):
            # Calculate the area of each contour
        area = cv2.contourArea(c)
        if int(area) > 1000:
            cv2.putText(img,str(int(area)),(20,a*30), 1,1,(0,0,255),1)
            a+=1
        x,y,w,h = cv2.boundingRect(c)
        # Ignore contours that are too small or too large
        if area < 2000 or 6000 < area:
            continue

        # Draw each contour only for visualisation purposes
        cv2.drawContours(img, c, -1, (0, 0, 255), 2)
        #center = (x+w//2, y+h//2)
        #cv2.circle(img,center,1,(255,0,0),1)
        #cv2.line(img,(x+w//2,y),(x+w//2,y+h),(0,0,0),2)
        cv2.putText(img,str(int(w*h)),(400,a*30), 1,1,(0,0,255),1)
        a+=1
        #object_mask= area
        #line_mask = (x+w//2,y),(x+w//2,y+h)
        #print(np.logical_and( object_mask, line_mask ))
        #cv2.bitwise_and(area,)
        cv2.putText(img,str(area),((x,y)),1,2,(0,0,255),2)
        # Find the orientation of each shape
        #angle = getOrientation(c, img)
        #cv2.putText(img,str(int(angle)),((x,y)),1,2,(0,0,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #print(angle)


    cv2.imshow('Output Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.waitKey(0)

cv2.destroyAllWindows()
    
# Save the output image to the current directory
#cv.imwrite("output_img.jpg", img)