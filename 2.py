import cv2
from lib import *

img = cv2.imread("motor3.jpg")
img = cv2.resize(img,(640,480))
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
b1,b2 = adjust_parameter(img)
img_c = img.copy()
result(img_c,b1,b2)
cv2.imshow('Output Image', img_c)
cv2.waitKey(0)



# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret,frame = cap.read()
#     if ret:
#         frame = cv2.resize(frame,(640,480))
#         if cv2.waitKey(1) & 0xff == ord('a'):
#             img = frame.copy()
#             result_video(img)
#             cv2.imshow('result', img)
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break
#     else:
#         break
# cap.release()


cv2.destroyAllWindows()

