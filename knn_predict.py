import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import shutil

filename = 0
class knn_prediction:

    def __init__ (self):
        proto = "ML/deploy.prototxt.txt"
        caffmodel = "ML/res10_300x300_ssd_iter_140000.caffemodel"
        self.model_path = None
        self.knn_clf = ""

        self.confid = 0.7
        self.net = cv2.dnn.readNetFromCaffe(proto, caffmodel)
        self.distance_threshold = 0.6

    def set_model_path(self, model_path):
        self.model_path = str(model_path)+"_knn.clf"
        with open(self.model_path, 'rb') as f:
            self.knn_clf = pickle.load(f)

    def get_model_path(self):
        return self.model_path


    def predict (self, img):

        image = np.array(img)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()


        for i in range(0, detections.shape[2]):
            # print(detections.shape)
            confidence = detections[0, 0, i, 2]

            if confidence > self.confid:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                try:
                    faces_encodings = face_recognition.face_encodings(image[startY:endY, startX: endX])
                    #print(len(faces_encodings))
                    # Use the KNN model to find the best matches for the test face
                    closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=3)
                    are_matches = [closest_distances[0][0][0] <= self.distance_threshold]
                    # Predict classes and remove classifications that aren't within the threshold
                    for pred , rec in zip(self.knn_clf.predict(faces_encodings), are_matches):
                        if rec:
                            #print (pred)
                            print(confidence)
                            return pred
                        else:
                            #print ("unknown")
                            return "unknown"
                    #return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
                except Exception as e:
                    print ("No 128d returned")

                    global filename
                    filename += 1

                    file = str(filename) + ".jpg"
                    # identityName = os.path.basename(os.path.dirname(img)).replace(" ", "-").lower()
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    img_fold = os.path.join(BASE_DIR, "error images")

                    if not os.path.exists(img_fold):
                        os.makedirs(img_fold)

                    #
                    # identity_fold = os.path.join(img_fold, identityName)
                    # if not os.path.exists(identity_fold):
                    #     os.makedirs(identity_fold)
                    #
                    cv2.imwrite(os.path.join(img_fold,file), image[startY:endY, startX: endX])


        return "error"
