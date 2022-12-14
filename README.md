## Spaceship Titanic

 This repository holds an attempt to the Spaceship Titanic Kaggle Challenge
 
 ![image](https://user-images.githubusercontent.com/98277599/207728311-4f41b5e2-d232-4424-b236-3d137cd2be49.png)


## Overview

  * The aim of the spaceship challenge is to help rescue crews and retrieve the lost passengers, I have to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
  * We performed feature engineering, data visualization and used 3 classifiers that were logisitic regression, decision tree classifier and random forest classifier
  * Our best model was able to predict with 80% accuracy of how many students went missing
## Summary of Workdone

### Data

* Data:
  * The data was of the csv type.
  * We had test data and train data of the list of all passengers.
  * The size of the data is 1.24mb.
  
 ### test.df:
 * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
 * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
 * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
 * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
* Age - The age of the passenger.
* VIP - Whether the passenger has paid for special VIP service during the voyage.
* RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
* Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

### test.csv:

Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.


#### Preprocessing / Clean up

* Made a couple histograms to visualize the data
* Found the outliers
* Dropped null values
* Coverted categorical and bool values into int columns. 

#### Data Visualization

![image](https://user-images.githubusercontent.com/98277599/207662029-76b52b8c-7987-42d8-b55f-ffe32301936b.png)

The room service variable had a lot of outliers which we eliminate later. 

![image](https://user-images.githubusercontent.com/98277599/207668589-38e3914f-2e14-4c11-a94f-858458d61dd4.png)


### Problem Formulation

* Define: 
  * Input / Output : Train data with passnger data without information on whether they're missing or not
  * Models: I used logistic regression, decision tree and random forest classifier. I got the best results from the random forest classifier. 
  * Logistic regression : Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
  * DecisionTreeClassifer : A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.
  * RandomForestClassifier : Random forest classifier can be used to solve for regression or classification problems. The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample.

### Training

* Describe the training:
  * How you trained: I used sklearn modules and uploaded the classifiers. I used pandas, matplotlib, seaborn to organiza and arrange the data. 
  * It took about 15-20 minutes to train the data
  * No difficulties
  

### Performance Comparison

Logistic Regression
![image](https://user-images.githubusercontent.com/98277599/207664951-34a8077c-5c36-499b-b7e7-44cf43a5c5ea.png)

DecisionTreeClassifier
![image](https://user-images.githubusercontent.com/98277599/207665098-9cf6d889-654c-46e0-89b4-7040e4fdf4e8.png)

RandomForestClassifier
![image](https://user-images.githubusercontent.com/98277599/207665250-61b5d3c9-c5a8-4910-954b-367e87d10c09.png)


### Conclusions

* Random forest classifer worked better than logistic regression and decision tree classifier
### Future Work

* Building neural network models or utilizing deep learning techniques

## How to reproduce results

   * I used google colab as it was a group project with another classmate. We used google colab to collaborate and used python and libraries that are sklearn, pandas, numpy, matplotlib and seaborn.

### Overview of files in repository

  * FinalExamProject.ipynb: file with all the code for training models and testing accuracy.
  * FinalEXamProject.py : file with all the code for training models and testing accuracy.
  * test.csv: CSV file of test data 
  * train.csv: CSV file for training data
  * submission.csv: CSV file to submit the code


### Software Setup
Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, lightgbm, joblib, keras
Download seaborn in jupyter - pip install seaborn

### Data

The data can be trained here https://www.kaggle.com/competitions/spaceship-titanic/data


## Citations

[* Provide any references.](https://www.kaggle.com/code/strategos2/spaceship-titanic-classification/notebook)

