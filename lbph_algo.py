import os
import cv2
import sys
import PIL
from PIL import Image #python image library
import numpy as np
import pickle
import lbph_predict as lp
import importlib
import imp

at = lp.algorithm_test()

def testAlgo(image, DS_DIR):
    if at.get_ymlfile() is None:
        at.set_ymlfile(DS_DIR)

    if at.get_pickFile() is None:
        at.set_pickFile(DS_DIR)

    return at.lbph_pred(image)

def trainAlgo(imageArr,labelArr, DIR_NAME):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # img_dir = os.path.join(BASE_DIR,"images")
    # training_data = os.path.join(img_dir, "Training Image")

    proto = "ML/deploy.prototxt.txt"
    caffmodel = "ML/res10_300x300_ssd_iter_140000.caffemodel"
    confid = 0.7

    net = cv2.dnn.readNetFromCaffe(proto, caffmodel)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {} #dictionary
    y_labels = []
    x_train = []


    for x in range(len(imageArr)):
            if not labelArr[x] in label_ids:
                label_ids[labelArr[x]] = current_id
                current_id += 1
                id_ = label_ids[labelArr[x]]
                #print(label_ids)

            #pil_image = Image.open(path).convert("L") #grayscale
            try:

                image = np.array(imageArr[x])
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                net.setInput(blob)
                detections = net.forward()

                for i in range(0, detections.shape[2]):
                    #print(detections.shape)

                    confidence = detections[0, 0, i, 2]
                    if confidence > confid:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        #print(box)
                        (startX, startY, endX, endY) = box.astype("int")
                        roi = image[startY:endY, startX: endX]
                        #roi = roi.cvt
                        try:
                            if(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).shape):
                                x_train.append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) #why do this is to save multiple person.
                                y_labels.append(id_)
                        except Exception as e:
                            pass
                #print((str(labelArr[x]) + " " + str(current_id)), end = "", flush = True)
                sys.stdout.write("\r" + str(x+1) + " of " + str(len(imageArr)) + " has been processed");
                sys.stdout.flush()
            except Exception as e:
                print(e)

    x_train = normalize_img(x_train)
    x_train = resize(x_train)
    with open(str(DIR_NAME) + "_LBPH.pickle",'wb') as f: #wb = writing byte
        pickle.dump(label_ids, f)
    recognizer.train(x_train,np.array(y_labels))
    recognizer.write(str(DIR_NAME) +"_LBPH.yml")
    print("\n **LBPH Training Completed**\n")

def normalize_img(images):
    images_normalized = []
    for image in images:
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_normalized.append(cv2.equalizeHist(image))
        except Exception as e:
            pass
    return images_normalized

def resize(images, size = (100,100)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

        images_norm.append(image_norm)
    return images_norm

def main():
    pythonFile = "benchmarker"
    nameList = []
    nameList.clear()
    try:
        bm = importlib.import_module(pythonFile, ".")
        menu = True
        spam_info = imp.find_module(pythonFile)
        print("Import Benchmark successful")


        ######## Dataset 1-4###################

        # DS_DIR = "Dataset 4"
        # imageArr, labelArr, DS_DIR = bm.fetchTrainingData(DS_DIR)
        # print("Run Successful")
        # trainAlgo(imageArr, labelArr, DS_DIR)
        #
        # imgArr = bm.fetchTestQuestion()
        # #print(len(imgArr))
        # for i in range(len(imgArr)):
        #     name = testAlgo(imgArr[i], DS_DIR)
        #     nameList.append(name)
        #
        # bm.submitAnswer(nameList)


        ########## Dataset 5 ######################

        DS_DIR = "Dataset 5"
        imageArr, labelArr, DS = bm.fetchTrainingData(DS_DIR)
        print("Run Successful")
        trainAlgo(imageArr, labelArr, DS)

        imgArr = bm.fetchTestQuestion()
        #print(len(imgArr))
        for i in range(len(imgArr)):
            name = testAlgo(imgArr[i], DS)
            nameList.append(name)

        correctAns1, wrongAns1, acc1 = bm.submitAnswer(nameList)
        nameList.clear()
        ### Training for Mixed Training
        imageArr2, labelArr2, DS_DIR2 = bm.fetchTrainingData(DS_DIR)
        print("Run Successful 2")

        trainAlgo(imageArr2, labelArr2, DS_DIR2+"2")# +"2" is to save the file under different name. If u want to replace the previous file, you need not +"2"
        ### Fetching 2nd test - Angle
        imgArr2 = bm.fetchTestQuestion()

        for i in range(len(imgArr)):
            name = testAlgo(imgArr2[i], DS_DIR2+"2")
            nameList.append(name)

        correctAns2, wrongAns2, acc2 = bm.submitAnswer(nameList)
        nameList.clear()
        ### Fetching 3rd Test - Lighting
        imgArr3 = bm.fetchTestQuestion()
        # print(len(imgArr))
        for i in range(len(imgArr3)):
            name = testAlgo(imgArr3[i], DS_DIR2+"2")
            nameList.append(name)

        correctAns3, wrongAns3, acc3 = bm.submitAnswer(nameList)
        nameList.clear()
        print("================= Test 1 (Pure Faces) Results ========================")
        print ("No of correct answer: " + str(correctAns1))
        print ("No of wrong answer: " + str(wrongAns1))
        print ("Accuracy: " + str(acc1))

        print("")
        print("============= Test 2 (Faces of different angle) Results ==============")
        print ("No of correct answer: " + str(correctAns2))
        print ("No of wrong answer: " + str(wrongAns2))
        print ("Accuracy: " + str(acc2))

        print("")
        print("============= Test 3 (Faces of different lighting) Results ==============")
        print ("No of correct answer: " + str(correctAns3))
        print ("No of wrong answer: " + str(wrongAns3))
        print ("Accuracy: " + str(acc3))

    except Exception as e:
        print (e)

if __name__ == "__main__":
    main()
