import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('classifierLBPH.yml')
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(1)

while True:
    connected, image = camera.read()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFaces = face_detector.detectMultiScale(grayImage, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, width, height) in detectedFaces:
        faceImage = cv2.resize(grayImage[y:y + height, x:x + width], (width, height))
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
        id, confidence = recognizer.predict(faceImage)
        cv2.putText(image, "Person number: " + id, (x, y - 5), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
