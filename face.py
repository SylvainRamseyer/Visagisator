import cv2
import sys
import numpy as np

# Get user supplied values
imagePath = sys.argv[1]
cascPath =  "haarcascade_frontalface_default.xml"
fondPath = sys.argv[2]

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
    #Gray conversion
    face.append(gray[y:y+h, x:x+w])

cv2.imshow("Faces found", image)
cv2.waitKey(0)


for facx in face:
    offset = 0
    img = facx

    img2 = img
    cv2.imshow("Visage gray", img2)
    cv2.waitKey(0)



    hsv = cv2.cvtColor(fond, cv2.COLOR_BGR2HSV)

    #parse and modify each pixels
    height, width = img.shape
    out = np.zeros((height,width,3), np.uint8)
    for i in range(height):
        for j in range(width):
            out[i,j] = [hsv[offset+i,offset+j][0],hsv[offset+i,offset+j][1], hsv[offset+i,offset+j][2]*(img2[i,j]/300)]

    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    cv2.imshow("Visage intergration", out)
    cv2.waitKey(0)
