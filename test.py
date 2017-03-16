import cv2
import sys
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get user supplied values
    imagePath = sys.argv[1]
    cascPath = sys.argv[2]
    fondPath = sys.argv[3]

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    fond = cv2.imread(fondPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

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

    for facx in face:
        offset = 200;
        img = facx

        ret,thresh = cv2.threshold(img,127,255,0)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,9)
        im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,255,0), 3)

        hsv = cv2.cvtColor(fond, cv2.COLOR_BGR2HSV)

        height, width = img.shape
        out = np.zeros((height,width,3), np.uint8)
        for i in range(height):
            for j in range(width):
                out[i,j] = hsv[offset+i, offset+j] if img[i,j] > 3 else [hsv[offset+i,offset+j][0],hsv[offset+i,offset+j][1], hsv[offset+i,offset+j][2]*0.4]#[img[i,j], img[i,j], img[i,j]]
        #img[0:height, 0:width] = fond[offset:height+offset, offset:width+offset] - img[0:height, 0:width]
        out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
        cv2.imshow("trellele", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
