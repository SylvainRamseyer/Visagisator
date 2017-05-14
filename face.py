import cv2
import sys
import numpy as np

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]
fondPath = sys.argv[3]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

fond = cv2.imread(fondPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

face = []

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face.append(gray[y:y+h, x:x+w])

cv2.imshow("Faces found", image)
cv2.waitKey(0)


for facx in face:
    offset = 0
    img = facx

    img2 = img
    cv2.imshow("img2", img2)
    cv2.waitKey(0)

    # ret,thresh = cv2.threshold(img,127,255,0)
    # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,8)
    # cv2.imshow("2", th2)
    # cv2.waitKey(0)
    # im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)


    hsv = cv2.cvtColor(fond, cv2.COLOR_BGR2HSV)

    # cv2.imshow("3", th2)
    # cv2.waitKey(0)

    height, width = img.shape
    out = np.zeros((height,width,3), np.uint8)
    for i in range(height):
        for j in range(width):
            out[i,j] = [hsv[offset+i,offset+j][0],hsv[offset+i,offset+j][1], hsv[offset+i,offset+j][2]*(img2[i,j]/300)]
            # out[i,j] = [hsv[offset+i,offset+j][0],hsv[offset+i,offset+j][1], hsv[offset+i,offset+j][2]+img2[i,j]*0.30]
            #out[i,j] = hsv[offset+i, offset+j] if th2[i,j] < 3 else [hsv[offset+i,offset+j][0],hsv[offset+i,offset+j][1], hsv[offset+i,offset+j][2]*0.5]#[img[i,j], img[i,j], img[i,j]]
    #img[0:height, 0:width] = fond[offset:height+offset, offset:width+offset] - img[0:height, 0:width]
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    cv2.imshow("trellele", out)
    cv2.waitKey(0)
