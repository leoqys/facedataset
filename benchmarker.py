import os
import random
import cv2
from PIL import Image
import numpy as np
import shutil
import importlib
import imp

testset_amount = 10
selected_Dataset = None
trainingSeq = 0
testSeq = 0
test_Question = []
test_Dir = []
bm = None
MajorDataset = "Dataset 5"

def fetchTrainingData(DS_DIR, TRAIN_DIR = "Training Image"):
    if DS_DIR == MajorDataset and TRAIN_DIR == "Training Image":
        global trainingSeq
        if trainingSeq == 0:
            trainingSeq = 1
            return fetchTrain(DS_DIR, "Training-Pure")

        elif trainingSeq == 1:
            trainingSeq = 2
            return fetchTrain(DS_DIR, "Training-Mix")

    else:
        return fetchTrain(DS_DIR, TRAIN_DIR)

def fetchTestQuestion(test_Name = "Test Image"):
    if selected_Dataset == MajorDataset and test_Name == "Test Image":
        global testSeq
        if testSeq == 0:
            testArr = fetchTest("Test 1")
            testSeq = 1
            return testArr

        elif testSeq == 1:
            testSeq = 2
            return fetchTest("Test 2 - Angle")

        elif testSeq == 2:
            testSeq = 3
            return fetchTest("Test 3 - Lighting")
    else:
        return fetchTest(test_Name)

def fetchTest(test_Name = "Test Image"):
    global test_Question
    global test_Dir

    if selected_Dataset is not None:
        DS_DIR = selected_Dataset

        test_Dir.clear()
        test_Question.clear()
        test_set = []

        print("** Fetching Test Images... **")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(BASE_DIR, DS_DIR)
        testset_data = os.path.join(img_dir, test_Name)

        total_files = len(os.listdir(testset_data)) * testset_amount
        rand = random.sample(range(0, total_files), total_files)

        for ran in rand:
            iden_id = int(ran / testset_amount)
            img_no = ran % testset_amount

            iden_dir = os.path.join(testset_data, os.listdir(testset_data)[iden_id])
            img = os.listdir(iden_dir)[img_no]
            testimg_dir = os.path.join(iden_dir, img)
            image = cv2.imread(testimg_dir)
            test_Dir.append(testimg_dir)
            test_set.append(image)
            test_Question.append(os.listdir(testset_data)[iden_id].replace(" ", "-").lower())
            #name = bm.testAlgo(image, DS_DIR)

        print("** Fetch Completed **")
        return test_set


def submitAnswer(ansArr):
    global test_Question
    global test_Dir
    correctAns = 0
    wrongAns = 0

    if not test_Question:
        print("Please fetch the test questions before submitting")
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        wrong_dir = os.path.join(BASE_DIR, "Incorrect Answer")

        if not os.path.exists(wrong_dir):
            os.makedirs(wrong_dir)
        else:
            shutil.rmtree(wrong_dir)
            os.makedirs(wrong_dir)

        print("** Checking your answer **")
        for x in range(len(test_Question)):
            print("Question: " + test_Question[x].replace(" ", "-").lower())
            if ansArr[x] is not None:
                print("Answer: " + ansArr[x].replace(" ", "-").lower())
            else:
                print("Answer: ")
            print("")
            if ansArr[x] is not None and ansArr[x].replace(" ", "-").lower() == test_Question[x]:
                correctAns += 1
            else:
                wrongAns += 1
                copyDir = str(x)  + " Qn-" + test_Question[x] + " Ans-" + ansArr[x].replace(" ", "-").lower()
                shutil.copyfile(test_Dir[x], (wrong_dir + "\\" + str(copyDir)))

    print("No of correct: " + str(correctAns))
    print("No of wrong: " + str(wrongAns))
    acc = (correctAns / (correctAns + wrongAns)) * 100
    print("Accuracy is " + str(acc) + "%\n")
    return correctAns, wrongAns, acc

def feedTestData(DS_DIR, TEST_DIR="Test Image"):
    print("**Initiating Test, calling testAlgo() method **")
    try:
        correctAns = 0
        wrongAns = 0
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(BASE_DIR, DS_DIR)
        testset_data = os.path.join(img_dir, TEST_DIR)
        wrong_dir = os.path.join(BASE_DIR,"Incorrect Answer")

        if not os.path.exists(wrong_dir):
            os.makedirs(wrong_dir)
        else:
            shutil.rmtree(wrong_dir)
            os.makedirs(wrong_dir)

        print("Total Number of test = " +str(len(os.listdir(testset_data))))
        total_files = len(os.listdir(testset_data)) * testset_amount
        rand = random.sample(range(0, total_files), total_files)

        for ran in rand:
            iden_id = int(ran / testset_amount)
            img_no = ran % testset_amount

            iden_dir = os.path.join(testset_data, os.listdir(testset_data)[iden_id])
            img = os.listdir(iden_dir)[img_no]
            testimg_dir = os.path.join(iden_dir, img)
            image = cv2.imread(testimg_dir)
            name = bm.testAlgo(image, DS_DIR)
            print("Question: " + os.listdir(testset_data)[iden_id].replace(" ", "-").lower())
            if name is not None:
                print("Answer: " + name.replace(" ", "-").lower())
            else:
                print("Answer: ")
            print("")
            if name is not None and name.replace(" ", "-").lower() == os.listdir(testset_data)[iden_id].replace(" ", "-").lower():
                correctAns += 1
            else:
                wrongAns += 1
                shutil.copyfile(testimg_dir, (wrong_dir + "\\" + img))

        print("No of correct: " + str(correctAns))
        print("No of wrong: " + str(wrongAns))
        acc = (correctAns / (correctAns+wrongAns))*100
        print("Accuracy is " + str(acc) + "%\n")

        return correctAns, wrongAns, acc

    except Exception as e:
        print("Please ensure you code have a method name testAlgo(image)")
        print (e)



def fetchTrain(DS_DIR, TRAIN_DIR = "Training Image"):
    try:
        imageArr = []
        labelArr = []
        global selected_Dataset
        selected_Dataset = DS_DIR

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(BASE_DIR, DS_DIR)
        training_data = os.path.join(img_dir, TRAIN_DIR)

        print("Preparing images for training...")

        for root, dirs, files in os.walk(training_data):
            for file in files:
                if file.lower().endswith("png") or file.lower().endswith("jpg") or file.lower().endswith("jpeg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                    image = cv2.imread(path)
                    imageArr.append(image)
                    labelArr.append(label)

        #bm.testAlgo(imageArr)
        return imageArr, labelArr, DS_DIR
        #bm.trainAlgo(imageArr,labelArr ,DS_DIR)
    except Exception as e:
        print(e)
        #print("Please ensure you code have a method name trainAlgo(imageArray[], label[], DS_NAME)")

        return None


def main():
    pythonFile = input("Key in your ALGORITHM file name (without .py): ")
    try:
        global bm
        bm = importlib.import_module(pythonFile, ".")
        menu = True
        spam_info = imp.find_module(pythonFile)
        #print(spam_info)
        print("Import ALGORITHM FILE successful")


        while True:
            print("Dataset to generate Train and Test Set:")
            print("1) Dataset 1")
            print("2) Dataset 2")
            print("3) Dataset 3")
            print("4) Dataset 4")
            print("5) Dataset 5")
            print("0) Exit")

            try:
                choice = int(input("Your Choice: "))
                if choice == 1:
                    print("**Initiating training, calling trainAlgo() method.**")
                    print("")
                    imageArr, labelArr, DS_DIR = fetchTrainingData("Dataset 1")
                    bm.trainAlgo(imageArr, labelArr, DS_DIR)
                    feedTestData("Dataset 1")
                elif choice == 2:
                    print("**Initiating training, calling trainAlgo() method.**")
                    print("")
                    imageArr, labelArr, DS_DIR = fetchTrainingData("Dataset 2")
                    bm.trainAlgo(imageArr, labelArr, DS_DIR)
                    feedTestData("Dataset 2")
                elif choice == 3:
                    print("**Initiating training, calling trainAlgo() method.**")
                    print("")
                    imageArr, labelArr, DS_DIR = fetchTrainingData("Dataset 3")
                    bm.trainAlgo(imageArr, labelArr, DS_DIR)
                    feedTestData("Dataset 3")
                elif choice == 4:
                    print("**Initiating training, calling trainAlgo() method.**")
                    print("")
                    imageArr, labelArr, DS_DIR = fetchTrainingData("Dataset 4")
                    bm.trainAlgo(imageArr, labelArr, DS_DIR)
                    feedTestData("Dataset 4")
                elif choice == 5:
                    print("**Initiating training(pure face images), calling trainAlgo() method.**")
                    print("")

                    imageArr, labelArr, DS_DIR = fetchTrainingData(MajorDataset, "Training-Pure")
                    bm.trainAlgo(imageArr, labelArr, DS_DIR)
                    print("Test 1 Started:")
                    correctAns, wrongAns, acc = feedTestData("Dataset Mix","Test 1")

                    print("\n Initiating training 2(mix factor images), calling trainAlgo() method")
                    imageArr2, labelArr2, DS_DIR2 = fetchTrainingData(MajorDataset, "Training-Mix")
                    bm.trainAlgo(imageArr2, labelArr2, DS_DIR2 + "2")

                    print("Test 2 Started: ")
                    correctAns2, wrongAns2, acc2 = feedTestData(MajorDataset, "Test 2 - Angle")
                    print("Test 3 Started: ")
                    correctAns3, wrongAns3, acc3 = feedTestData(MajorDataset, "Test 3 - Lighting")

                    print("================= Test 1 (Pure Faces) Results ========================")
                    print ("No of correct answer: " + str(correctAns))
                    print ("No of wrong answer: " + str(wrongAns))
                    print ("Accuracy: " + str(acc))

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

                elif choice == 0:
                    print("Exiting ...")
                    break
                else:
                    print("Invalid input.")
            except Exception as e:
                print(e)
                print("Please check you have selected the correct Dataset")


    except Exception as e:
        print(e)
        print("Fail to import ALGORITHM file. Please check that ")

if __name__ == "__main__":
    main()













