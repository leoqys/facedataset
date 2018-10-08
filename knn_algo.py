import importlib
import imp
import math
from sklearn import neighbors
import os
import sys
import pickle
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import knn_predict as fra

knp = fra.knn_prediction()
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# img_dir = os.path.join(BASE_DIR, "../images")
# training_data = os.path.join(img_dir, "Training Image")

# def testAlgo(img):
#     #pil_image = Image.open('Image.jpg').convert('RGB')
#     #print(img[3])
#     open_cv_image = np.array(img[5])
#     imageRGB = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
#     open_cv_image.shape[:2]
#     print(len(img))
#     cv2.imshow('image', open_cv_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
def testAlgo(image , DIR_NAME):

    if knp.get_model_path() is None:
        knp.set_model_path(DIR_NAME)

    return knp.predict(image)

def trainAlgo(imageArr,labelArr, DIR_NAME):

    X_train = []
    y_labels = []
    model_save_path = str(DIR_NAME) + "_knn.clf"
    n_neighbors = 3
    #model_save_path = None
    #n_neighbors = None
    knn_algo = 'ball_tree'
    verbose = False

    proto = "ML/deploy.prototxt.txt"
    caffmodel = "ML/res10_300x300_ssd_iter_140000.caffemodel"
    confid = 0.7

    net = cv2.dnn.readNetFromCaffe(proto, caffmodel)

    for x in range(len(imageArr)):
            #print("Training Identity " + labelArr[x] + " " + str(x))
            sys.stdout.write("\r" + str(x + 1) + " of " + str(len(imageArr)) + " has been processed");
            sys.stdout.flush()
            try:
                count = 0
                imageA = np.array(imageArr[x])
                #imageRGB = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
                (h, w) = imageA.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(imageA, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))


                net.setInput(blob)
                detections = net.forward()

                for i in range(0, detections.shape[2]):

                        # print(detections.shape)
                        count += 1
                        confidence = detections[0, 0, i, 2]
                        if confidence > confid and count == 1:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            #face_bounding_boxes = "("+startX+","+endX+","+startY+","+endY+")"
                            roi = imageA[startY:endY, startX: endX]
                            #print(face_recognition.face_encodings(roi))
                            X_train.append(face_recognition.face_encodings(roi)[0])
                            y_labels.append(labelArr[x])

            except Exception as e:
                print ("")
                print (e)

    if n_neighbors is None:
                n_neighbors = int(round(math.sqrt(len(X))))
                if verbose:
                    print("Chose n_neighbors automatically:", n_neighbors)

            # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X_train, y_labels)

            # Save the trained KNN classifier
    if model_save_path is not None:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(knn_clf, f)
                print("**Training Completed**")

    return knn_clf

def main():
    pythonFile = "benchmarker"
    nameList = []
    nameList.clear()
    try:
        bm = importlib.import_module(pythonFile, ".")
        menu = True
        spam_info = imp.find_module(pythonFile)
        print("Import Benchmark successful")

        DS_DIR = "Dataset 4"
        imageArr, labelArr, DS_DIR = bm.fetchTrainingData(DS_DIR)
        print("Run Successful")
        trainAlgo(imageArr, labelArr, DS_DIR)

        imgArr = bm.fetchTestQuestion()
        #print(len(imgArr))
        for i in range(len(imgArr)):
            name = testAlgo(imgArr[i], DS_DIR)
            nameList.append(name)

        bm.submitAnswer(nameList)

    except Exception as e:
        print (e)

if __name__ == "__main__":
    main()

