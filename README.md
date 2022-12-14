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
  
 #### train.csv:
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

#### test.csv:

Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.


### Preprocessing / Clean up

Firstly, we made a couple of histrograms to better understand the data. After that we got rid of the null values and and converted categorical values and bool values into int values. We performed feature engineering using SimpleImputer from sklearn.impute.

![image](https://user-images.githubusercontent.com/98277599/207733424-dcda6ef8-c2d7-45c9-af11-c2f0bad077c8.png)

Figure 1: This shows the number of null values for each variable

![image](https://user-images.githubusercontent.com/98277599/207733752-1f522004-f2c9-47f4-894a-32d6380532c5.png)

Figure 2: Converting to integer columns.

#### Data Visualization

![image](https://user-images.githubusercontent.com/98277599/207734027-9b234f8b-03d9-45f4-bf1a-8935a6a9a48a.png)

Figure 3: Passngers from their homeplanet and whether or not they were transported.

![image](https://user-images.githubusercontent.com/98277599/207734337-47e204e6-7b52-4eaf-a89f-d7d63bab25f7.png)

Figure 4: Representation of the age of the passengers.

![image](https://user-images.githubusercontent.com/98277599/207734393-2880c58d-debb-4ee9-af00-2a039d7c232b.png)

Figure 5: Outliers in the category of roomservice are visible here.


### Problem Formulation

* Define: 
  * Input / Output : The model is trained using the train data and then provided with the testing data where it predicts which passngers got transported.
  * Models: I used logistic regression, decision tree and random forest classifier. I got the best results from the random forest classifier. 
   #### Logistic regression : Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
   #### DecisionTreeClassifer : A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.
  #### RandomForestClassifier : Random forest classifier can be used to solve for regression or classification problems. The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample.

### Training

* Describe the training:
  * How you trained: I used sklearn modules and uploaded the classifiers. I used pandas, matplotlib, seaborn to organiza and arrange the data. 
  * It took about 15-20 minutes to train the data
  * No difficulties
  
  
Logistic Regression
![image](https://user-images.githubusercontent.com/98277599/207664951-34a8077c-5c36-499b-b7e7-44cf43a5c5ea.png)

DecisionTreeClassifier
![image](https://user-images.githubusercontent.com/98277599/207665098-9cf6d889-654c-46e0-89b4-7040e4fdf4e8.png)

RandomForestClassifier
![image](https://user-images.githubusercontent.com/98277599/207665250-61b5d3c9-c5a8-4910-954b-367e87d10c09.png)

### Performance Comparison

As mentioned earlier, random forest classifier performed the best of them all.

![image](https://user-images.githubusercontent.com/98277599/207734222-872940e0-b861-433f-9acd-d6c0f6063445.png)


### Conclusions

* Random forest classifer worked better than logistic regression and decision tree classifier
### Future Work

* Building neural network models or utilizing deep learning techniques.

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

