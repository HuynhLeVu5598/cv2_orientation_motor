from tabnanny import check
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
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    #anpha = 0.02
    #p1 = (cntr[0] + anpha * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + anpha * eigenvectors[0,1] * eigenvalues[0,0])
    #print(p1)
    #p2 = (cntr[0] - anpha * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - anpha * eigenvectors[1,1] * eigenvalues[1,0])

    #drawAxis(img, cntr, p1, (255, 255, 0), 1)
    #drawAxis(img, cntr, p2, (0, 0, 255), 1)
    
    # atan2 (y, x) trả về góc θ giữa tia tới điểm (x, y) và trục x dương, giới hạn trong (−π, π]
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    angle = angle*180/3.14
    return angle,cntr

def convert_img(img,b1=176,b2=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, b1, b2,  cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return bw,contours

def transform1(img,contours,min_area=6000,max_area=12000):
    #list_area = []
    list_cntr = []
    list_angle = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if area < min_area or max_area < area:
            continue
        angle,cntr = getOrientation(c,img)
        list_angle.append(angle)
        list_cntr.append(cntr)

        if -20 < angle <20 or 160 < angle < -160:
            center_x,center_y = (x+w//2,y),(x+w//2,y+h)
        if 70 <angle < 110 or -110 < angle < -70:
            center_x,center_y = (x,y+h//2),(x+w,y+h//2)
        cv2.line(img,center_x, center_y,(0,0,0),1)
    return list_angle,list_cntr

def transform2(img,contours,min_area=2000,max_area=6000):
    list_area = []
    list_x = []
    list_y = []
    list_w = []
    list_h = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if area < min_area or max_area < area:
            continue
        list_area.append(area)
        list_x.append(x)
        list_y.append(y)
        list_w.append(w)
        list_h.append(h)
        cv2.drawContours(img, c, -1, (0, 0, 255), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return list_area,list_x, list_y, list_w, list_h

def adjust_parameter(img):
    def nothing(x):
        pass
    cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
    cv2.createTrackbar('b1','image',176,255,nothing)
    cv2.createTrackbar('b2','image',255,255,nothing)
    while True:
        b1 = cv2.getTrackbarPos('b1','image')
        b2 = cv2.getTrackbarPos('b2','image')
        #img = cv2.imread(name_image)
        img = cv2.resize(img,(640,480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, b1, b2,  cv2.THRESH_BINARY)
        
        cv2.imshow('bw',bw)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break

    cv2.destroyAllWindows()
    return b1,b2

def result(img,b1=176,b2=255):
    _,contours1 = convert_img(img,b1,b2)
    angle,cntr = transform1(img,contours1,4000,40000)
    bw2,contours2 = convert_img(img)
    cv2.imshow('bw',bw2)
    area2,x2,y2,w2,h2 = transform2(img,contours2,2000, 20000)

    for i in range(len(area2)):
        if i%2==0:
            cv2.circle(img,cntr[i//2],4,(255,0,0),-1)
            try:
                if area2[i] < area2[i+1]:
                    center_x = x2[i]+w2[i]//2
                    center_y = y2[i]+h2[i]//2 
                    center = (center_x,center_y)
                    cv2.circle(img, center,4,(255,0,0),-1)
                else:
                    center_x = x2[i+1]+w2[i+1]//2
                    center_y = y2[i+1]+h2[i+1]//2 
                    center = (center_x,center_y)
                    cv2.circle(img, center,4,(255,0,0),-1)
                center_o = cntr[i//2]
                if -20 < angle[i//2] <20 or 160 < angle[i//2] < -160:
                    if cntr[i//2][0] < center_x:
                        cv2.putText(img,' right',(center_o[0],center_o[1]),1,2,(255,255,0),2)
                    if cntr[i//2][0] > center_x:
                        cv2.putText(img,' left' ,(center_o[0],center_o[1]),1,2,(255,255,0),2)
                if 70 <angle[i//2] < 110 or -110 < angle[i//2] < -70:
                    if cntr[i//2][1] < center_y:
                        cv2.putText(img,' down',(center_o[0],center_o[1]),1,2,(255,255,0),2)
                    if cntr[i//2][1] > center_y:
                        cv2.putText(img,' up',(center_o[0],center_o[1]),1,2,(255,255,0),2)
            except:
                print('error')

def result2(img):
    _,contours1 = convert_img(img)
    angle,cntr = transform1(img,contours1)
    bw2,contours2 = convert_img(img)
    cv2.imshow('bw',bw2)
    area2,x2,y2,w2,h2 = transform2(img,contours2)

    for i in range(len(area2)):
        if i%2==0:
            center_o = cntr[i//2]
            cv2.circle(img,cntr[i//2],4,(255,0,0),-1)
            if area2[i] < area2[i+1]:
                center_x = x2[i]+w2[i]//2
                center_y = y2[i]+h2[i]//2 
                center_b = (center_x,center_y)
                cv2.circle(img, center_b,4,(255,0,0),-1)
            else:
                center_x = x2[i+1]+w2[i+1]//2
                center_y = y2[i+1]+h2[i+1]//2 
                center_b = (center_x,center_y)
                cv2.circle(img, center_b,4,(255,0,0),-1)

            cv2.line(img,center_o, center_b,(255,255,0),2)
            #angle = atan2()
            print(str(int(angle)))

# center : center bounding
# cntr   : center object


def adjust_parameter_video(img):
    def nothing(x):
        pass
    cv2.namedWindow('image',cv2.WINDOW_FREERATIO)
    cv2.createTrackbar('b1','image',176,255,nothing)
    cv2.createTrackbar('b2','image',255,255,nothing)
    while True:
        b1 = cv2.getTrackbarPos('b1','image')
        b2 = cv2.getTrackbarPos('b2','image')
        #img = cv2.imread(name_image)
        #img = cv2.resize(img,(640,480))
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, b1, b2,  cv2.THRESH_BINARY)
        
        cv2.imshow('bw',bw)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break

    cv2.destroyAllWindows()
    return b1,b2

def result_video(img):
    b1,b2= adjust_parameter_video(img)
    _,contours1 = convert_img(img,b1,b2)
    angle,cntr = transform1(img,contours1,2000,40000)
    bw2,contours2 = convert_img(img)
    cv2.imshow('bw',bw2)
    area2,x2,y2,w2,h2 = transform2(img,contours2,2000, 20000)

    for i in range(len(area2)):
        if i%2==0:
            cv2.circle(img,cntr[i//2],4,(255,0,0),-1)
            if area2[i] < area2[i+1]:
                center_x = x2[i]+w2[i]//2
                center_y = y2[i]+h2[i]//2 
                center = (center_x,center_y)
                cv2.circle(img, center,4,(255,0,0),-1)
            else:
                center_x = x2[i+1]+w2[i+1]//2
                center_y = y2[i+1]+h2[i+1]//2 
                center = (center_x,center_y)
                cv2.circle(img, center,4,(255,0,0),-1)
            center_o = cntr[i//2]
            if -20 < angle[i//2] <20 or 160 < angle[i//2] < -160:
                if cntr[i//2][0] < center_x:
                    cv2.putText(img,' right',(center_o[0],center_o[1]),1,2,(255,255,0),2)
                if cntr[i//2][0] > center_x:
                    cv2.putText(img,' left' ,(center_o[0],center_o[1]),1,2,(255,255,0),2)
            if 70 <angle[i//2] < 110 or -110 < angle[i//2] < -70:
                if cntr[i//2][1] < center_y:
                    cv2.putText(img,' down',(center_o[0],center_o[1]),1,2,(255,255,0),2)
                if cntr[i//2][1] > center_y:
                    cv2.putText(img,' up',(center_o[0],center_o[1]),1,2,(255,255,0),2)