# Project 1

## Scheme

* data /
   * test.csv 
   * train.csv  
   * preprocessed_test.csv 
   * preprocessed_train.csv
* src /
  * main.py - main code to run datapreprocessing, create model and make predictions
  * model_params.yaml - contains hyperparameters, file paths and model name
  * Logistic.py  
  * data_preprocessing.py 
* results /
  * predicted.csv 
  * results.csv - contains model name, accuracy, precision, recall, f1 score and hyperparameters



## Goal

The goal of this project is to predict if a passenger survived the sinking of the Titanic or not using Logistic Regression 



## Subtasks

* Create Logistic Regression class

* Preprocess train and test datasets

* Create a model fitting our train data

* Make prediction, save results

  

### Step 1 - Create Logistic Regression 

**Logistic.py** file contains class named LogisticRegression

*Attributes:*

1. x_train : 2-dim numpy array

   ​		array of features

2. y_train : 1-dim numpy array

   ​		array of labels

3. coef : 1-dim numpy array

   ​		coefficients of features

4. intercept : float

   ​		intercept (bias) of regression

5. iterations : int

   ​		number of iterations, used for optimizing parameters

6. learning_rate : float

   ​		also used for optimizing coefficients and intercept

*Methods:*

1. init

   ​	Parameters:

   ​		learning_rate : float   ( default is 0.05 )
   
   ​		iterations : int   ( default is 1000 )


2. fit

   ​	Trains the model

   ​	Parameters:

   ​        x : 2-dim numpy array    # features

   ​        y : 1-dim numpy array	# labels

3. sigmoid

   ​	Maps real values into another value within a range of 0 and 1 

   ​	Parameters:

   ​		a : float  

4. gradient_descent

   ​	Optimizes coefficients and intercept			

5. print_weights 

   ​	Prints coefficients and intercept of the model

6. print_df

   ​	Prints train dataframe

7. predict

   ​	Predicts labels

   ​	Parameters:

   ​		x : 2-dim numpy array   #array of features

8. score

   ​	Returns the accuracy of the model

   ​	Parameters:

   ​		x_test : 2-dim numpy array   #array of features

   ​		y_test : 1-dim numpy array   #array of labels

9. predict_proba

   ​		Returns probabilities for each sample
   
   ​		Parameters:
   
   ​			x_test : 2-dim numpy array   #array of features
   
9. confusion_matrix

   ​		Returns list of True Positive, True Negative, False Positive, False Negative predictions
   
   ​		Parameters:
   
   ​			y_test : 1-dim numpy array   #array of true labels
   
   ​			y_pred : 1-dim numpy array   #array of predicted labels



## Step 2 - Preprocess train and test datasets

**data_preprocessing.py** contains following steps to preprocess train and test datasets:

1. Preprocess

​		Preprocesses datasets using coresponding functions

​		Arguments:

​			train : str   #path of train_df  

​			test : str #path of test_df

​		Returns:

​			preprocessed train and test datasets

1. isAlone

   ​	Adds a column 'isAlone' based on columns 'SibSp' and 'Parch'

   ​	Arguments:

   ​		df : pandas Dataframe

2. drop_

   ​	Drops unnecessary features such as Ticket, Name, Cabin, Parch, SibSp

   ​	Arguments:

   ​		df : pandas Dataframe

3. fill_

   ​	Fills missing cells of Fare and Embarked

   ​	Arguments:

   ​		df : pandas Dataframe   #df to be filled

   ​		base_df : pandas Dataframe    #based on which df must be filled 

4. map_

   ​	Converts categorical features, containing strings to numerical values

   ​	Arguments:

   ​		df : pandas Dataframe

5. fill_age

   ​	Fills missing values of Age feature

   ​	Arguments:

   ​		df : pandas Dataframe   #df to be filled

   ​		base_df : pandas Dataframe    #based on which df must be filled 

6. age_banding

   ​	Converts the Age feature to ordinal values based on bandes

   ​	Arguments:

   ​		df : pandas Dataframe

7. fare_banding

   ​	Converts the Fare feature to ordinal values based on bandes

   ​	Arguments:

   ​		df : pandas Dataframe



## Step 3, 4 - Create a model, make predictions and save results

**main.py** file contains following functions:

1. read_params

   ​	Create argument parser and reads parameters from given path

   ​	Returns:

   ​		params : dictionary with model parameters

2. prepare_data

   ​	Prepares data by spliting and converting to numpy array

   ​	Arguments:

   ​		train_df : pandas DataFrame

   ​		test_df : pandas DataFrame

   ​	Returns:

   ​		x_train : 2-dim numpy array 

   ​		x_test : 2-dim nupy array

   ​		y_train : 1-dim numpy array

    	   y_test : 1-dim numpy array

3. predict_save 

   ​	Make predictions and save ready dataset to csv file

   ​	 Arguments:

   ​		test_df : pandas DataFrame

   ​		model : LogisticRegression obj

   ​		path : str   #path to predicted.csv file

4. save_results

   ​	Save model parameters with result scores to given path

   ​	Arguments:

   ​		y_true : 1-dim numpy array 

   ​		y_predicted : 1-dim numpy array

   ​		params : dictionary   #containg model parameters

5. main function:

   * Reads parameters with read_params()
   * Preprocesses data with data_preprocessing.py
   * Prepares features and labels with prepare_data()
   * Creates LogisticRegression object with Logistic.py
   * Trains the model, make predictions, calculate accuracy
   * Saves predictions and results with predict_save() and save_results

   ​	

   

   ​	















