import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


class algorithm_test:

    def __init__(self):
        caffmodel = "ML/res10_300x300_ssd_iter_140000.caffemodel"
        proto = "ML/deploy.prototxt.txt"
        self.confid = 0.7
        self.net = cv2.dnn.readNetFromCaffe(proto, caffmodel)
        self.ymlfile = None
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.pickFile = None


    def set_ymlfile(self, ymlfile):
        self.ymlfile = str(ymlfile)+"_LBPH.yml"
        self.recognizer.read(self.ymlfile)

    def set_pickFile(self, pickFile):
        self.pickFile = str(pickFile)+"_LBPH.pickle"
        with open(self.pickFile, 'rb') as f:
            self.labels = pickle.load(f)
            #print(self.labels)
            self.labels = {v:k for k,v in self.labels.items()}

    def get_pickFile(self):
        return self.pickFile

    def get_ymlfile(self):
        return self.ymlfile


    def lbph_pred(self, img):
        name = ""
        roi_gray = []

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
                    (cv2.cvtColor(image[startY:endY, startX: endX], cv2.COLOR_BGR2GRAY))
                    roi_gray.append(cv2.cvtColor(image[startY:endY, startX: endX], cv2.COLOR_BGR2GRAY))

                    roi_gray = self.normalize_img(roi_gray)
                    roi_gray = self.resize(roi_gray)

                    id_, conf = self.recognizer.predict(roi_gray[0])
                    print("confidence: " + str(conf))
                    if conf >= 0 and conf <= 120:
                        name = self.labels[id_]
                        #print(name)
                        #print(self.labels[id_])
                except Exception as e:
                    pass

        return name

    def normalize_img(self, images):
        images_normalized = []
        for image in images:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_normalized.append(cv2.equalizeHist(image))
        return images_normalized

    def resize(self, images, size=(100, 100)):
        images_norm = []
        for image in images:
            if image.shape < size:
                image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            else:
                image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

            images_norm.append(image_norm)
        return images_norm

    # def lbphtest(self,img):
    #     roi_gray = []
    #     #gray = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    #
    #     image = cv2.imread(img)
    #     (h, w) = image.shape[:2]
    #     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    #
    #     self.net.setInput(blob)
    #     detections = self.net.forward()
    #
    #     for i in range(0, detections.shape[2]):
    #         # print(detections.shape)
    #         confidence = detections[0, 0, i, 2]
    #
    #         if confidence > self.confid:
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #
    #             roi_gray.append(cv2.cvtColor(image[startY:endY, startX: endX], cv2.COLOR_BGR2GRAY))
    #
    #             roi_gray = self.normalize_img(roi_gray)
    #             roi_gray = self.resize(roi_gray)
    #
    #             id_, conf = self.recognizer.predict(roi_gray[0])
    #             print(id_ , conf)
    #             if conf >=0 and conf<= 140:
    #             #if  conf <= 140:
    #                 # print(id_)
    #                 # print(labels[id_])
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 name = self.labels[id_]
    #                 color = (255,255,255)
    #                 stroke = 2
    #                 cv2.putText(gray,name,(startX,startY),font,1,color,stroke, cv2.LINE_AA)
    #                 print(name)
    #                 print("tadah")
    #
    #             img_item = "my-image.png"
    #             cv2.imwrite(img_item,roi_gray[0])
    #
    #             color = (255,0,0) #in BGR instead of RGB
    #             width = x+w # end coord
    #             height = y+h #end coord
    #             stroke = 2
    #             cv2.rectangle(gray, (x,y), (width,height), color,stroke)
    #
    #     cv2.imshow('frame', gray)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


#
# test = algorithm_test()
# test.lbphtest()