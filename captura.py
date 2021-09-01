import cv2
import numpy as np

# sets the classifier to use the xml code provided by the course
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# sets the eye classifier to use the xml code provided by the course
eyeClassifier = cv2.CascadeClassifier("haarcascade_eye.xml")

# set the camera from the device to be used in the application, being 0 the camera of the device
camera = cv2.VideoCapture(0)
# save the number of the sample (to be incremented each time the person captures its own photos)
sample = 1
# the total samples to be captured
totalSamples = 30
person_id = input('Type your id and press enter: ')
width, height = 220, 220

font = cv2.FONT_HERSHEY_SIMPLEX

print("Capturing faces...")

while True:
    # connect the camera and read the images
    connected, image = camera.read()

    # as the face detector runs better with gray images, turn the images into gray scale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(np.average(grayImage))

    # to keep all the faces detected with the camera
    # reminder> facesDetected returns:
    #  - an matrix with x and y as the starting points of the face
    #  - w and h as the height and width of the face
    facesDetected = classifier.detectMultiScale(grayImage, scaleFactor=1.5, minSize=(150, 150))

    # to display the brightness of the picture while the camera is open
    brightness = np.average(grayImage)
    cv2.putText(image, "Brightness: " + str(brightness), (0, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "PHOTO N " + str(sample), (0, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # to render the rectangle around the face
    for (x, y, w, h) in facesDetected:
        # cv2.putText(image, "Person number: " + person_id, (x, y - 5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # detect the eyes based on the face
        faceRegion = image[y:y + h, x:x + w]
        faceRegionGrayScale = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2GRAY)
        eyesDetected = eyeClassifier.detectMultiScale(faceRegionGrayScale)

        # draw an rectangle for the eyes and wait for the Q key to be pressed to save the picture
        for (eye_x, eye_y, eye_w, eye_h) in eyesDetected:

            cv2.rectangle(faceRegion, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # get the average brightness of the image and compare with the base to only capture pictures that has more than 110
                if np.average(grayImage) > 110:
                    imageFace = cv2.resize(grayImage[y:y + h, x:x + w], (width, height))
                    cv2.imwrite("photos/person." + str(person_id) + "." + str(sample) + ".jpg", imageFace)
                    sample += 1
                    cv2.waitKey(100)



    # sets the Face as the name of the window and pass the image to be render
    cv2.imshow("Face", image)
    cv2.waitKey(1)
    if sample >= totalSamples + 1:
        break

print("Faces captured with success")
camera.release()
cv2.destroyAllWindows()
