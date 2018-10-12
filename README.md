
The objective of this project is to create a meaningful dataset to help understanding of face recognition algorithms
There are two ways of using the benchmarker. 


Method 1: The benchmarker passes data by calling the functions from the algorithm file.
		
		The functions required by the algorithm are:
			- trainAlgo(imageArr, labelArr, DIR_NAME)
					This method is used to collect the entire training data with label names. It is used to train the algorithm.
					
				    Parameters: imageArr - A list of images (each as numpy array)
								labelArr - A list of label names (Associated with imageArr)
								DIR_NAME - Name of Dataset Folder (This is for algorithms to know which dataset it is using. This is optional for algorithms if they want to use this parameter for processing.)

			- testAlgo(image, DIR_NAME)
					This function is for the benchmarker to test the algorithm. The benchmarker will repeatedly challenge the algorithm by calling this function. A return of the identity name should be returned to the benchmarker.
					
					Parameters: image - a single image(In numpy array)
								DIR_NAME - Name of the Dataset Folder
								
					Expected Returns: Label name of the image.

		The user need to execute benchmarker.py and input the algorithm filename (without .py). The user will than need to have the above 2 functions for training and testing the algorithm.


		
Method 2: Algorithm to call the benchmarker to request for data for training and testing.

			The functions are to be called for training and testing:
				- fetchTrainingData(DS_DIR)
						This function is to request for training set from the benchmarker.
						
						Parameters: DS_DIR - Name of Dataset Folder (required for benchmarker to know which Dataset to use.)
						
						Returns: imageArr - A list of images (each as numpy array) 
								 labelArr - A list of label names (Associated with imageArr)
								 DS_DIR - Name of Dataset Folder
								 
				- fetchTestQuestion()
						This function is to request for testing dataset from the benchmarker. The function needs to be executed after fetchTrainingData(DS_DIR).
						
						Returns: test_set - A list of images (each as numpy array)
						
				- submitAnswer(ansArr)
						This function is to submit answer of to the test questions and the benchmarker will compare with its answer sheet. This function have to be called after fetchTestQuestion()
						
						Parameters: ansArr - A list of label names associated with the array from fetchTestQuestion()
						
						Returns: correctAns - Number of correct recognition
								 wrongAns - Number of wrong recognition
								 acc - Accuracy
		
			The user have to add this 3 functions in to the algorithm to test the performance of algorithm of different Dataset. 
		
			!!!!! Instruction for Dataset 5
			Dataset 5 consist of 2 training phase and 3 test phase. The purpose of Dataset 5 is listed under DATASETS.
			The steps to call and train dataset 5 is as followed.
			
			fetchTrainingData(DS_DIR) -> fetchTestQuestion() -> submitAnswer(ansArr) -> 
			fetchTrainingData(DS_DIR) -> fetchTestQuestion() -> submitAnswer(ansArr) ->
			fetchTestQuestion() -> submitAnswer(ansArr)
		
			The first set of <<fetchTrainingData(DS_DIR) -> fetchTestQuestion() -> submitAnswer(ansArr)>> is to test the algorithm when detecting pure faces. 
			The 2nd fetchTrainingData(DS_DIR) is to train the algorithm with different angle of faces.
			the 2nd and 3rd fetchTestQuestion is to test the algorithm with the respective purpose of the dataset.
********************************************************* DATASETS ************************************************************************************		
		
	There are currently 4 Datasets available. Currently, Datasets are a subset of FaceScrub Dataset.
	
	Dataset 1:
	This Dataset consist of images with single face extracted using Haar Cascade/Viola Jones Detection algorithm. It consist of 6,020 images with 50 identities. 
	By default, 50 images will be used for training and 10 images for testing per identity. 47 of the 50 identities will have sufficient images for benchmarking.
	The number of images for training and testing can be adjusted be regenerating the dataset using TrainTest_Generator.py.
	
	Dataset 2:
	This Dataset consist of images with single face extracted using Deep Learning detection (One-Shot Detector). It consist of 6,925 images with 50 identities.
	Similarly, 50 images will be used for training and 10 images for testing per identity. 49 of 50 identities will have sufficient images for benchmarking.
	
	Dataset 3:
	This dataset consist of images of single frontal face with consistent lighting throughout the entire face. Faces are not covered by any objects and is consisted "Clean Face".
	This dataset includes 10 males and 10 females with at least 50 images for each identity with 1,310 images in total. 40 images and 10 images will be used for training and testing respectively.
	
	Dataset 4:
	This dataset consist of 2,465 images with people with darker skin tone. This is because facial recognition algorithms have trouble detecting faces for with darker skin tone. 
	50 images and 10 images are used for training and testing respectively with 21 identities. 

	Dataset 5:
	This dataset consist of 3 test and 2 training sets. The first training set is of pure faces and it consist of 800 training image. This first trainingset if for the first test. It is to test the performance of 
	algorithm when recognising "pure" faces. 
	The 2nd training set is for test 2 and test 3. Training set 2 consist of 1000 training images with faces of different angle. Test set 2 is to test the performance of algorithms when detecting different angle of faces
	while test set 3 is for performance of different lighting.
	
	
 ******* The dataset is still building in progress. More data is needed. *******
 