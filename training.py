import cv2
import os
import numpy as np

# import of the algorithms to use the face recognizer
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImageWithId():
    # get the images from the photos folder and then convert those to gray scale before the training
    paths = [os.path.join('photos', p) for p in os.listdir('photos')]
    faces = []
    ids = []
    for imagePath in paths:
        image_face = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        ids.append(id)
        faces.append(image_face)
        # cv2.imshow("Face", image_face)
        # cv2.waitKey(250)
    return np.array(ids), faces


ids, faces = getImageWithId()
# print(faces)

print('training...')
eigenface.train(faces, ids)
eigenface.write('classifierEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classifierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifierLbph.yml')

print('Finished ')
