# Water pump functionality prediction &amp; analysis
# ![alt text](https://github.com/Eucalyptusss/Project_3/blob/main/images/readmewp.jpg)
## Project Overview
This repository analyzed the data at https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/.
The goal is to build a classification machine learning model that can accurately predict if a water pump is functional or not.
After trying many models the best two performing were the skleanrn's ensenmble methods RandomForest and a Gradient Boost. After
using techniques like gridsearch and random grid search I determine that the RandomForest model is the most accurate for this project.
After piplineing my process, I tested the final model on an unseen validation and finished with an accuracy score of .759. After validating
the final model I derived feature importances. From, there I found 3 conclusions that are relvant to the business problem.

## Business Problem
The government of Tanzania is obligated to provide accessible clean water to it's citizens. However, they are spending too much money for maintenance
& repair on the water sources. By determining the leading causes in a faulty water pump the government can prevent mechanical errors. Preventing the 
faulty pump phenomena will save the government money. Additionally, this data may produce results that are valuable to any organization that builds 
water pumps.

## Business Understanding
Taarifa has collected data regarding water sources in Tanzania. In order to save the government of Tanzania money I will determine which factors are 
most likely to cause water pump failure. To do this I will construct a machine learning model that will predict if a water source is faulty or not. 
Once I have confirmed my model is reliable, I will break down which independent variables are most correlated with faulty water sources. This will
allow water source providers to more about why their water pumps are failing.

## Data
All information regarding this data can be found at https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/

## Clean
### NaN
The data had 6 column with NaN values. The values were all small enough and categorical that I only had to replace NaN values with the string 'unknown'.
### DataTypes
There were 5 columns that were object but needed to be integers.
## Modeling
I cross validated various models using the train data. I used accuracy to determine the model of best fit.
- Logistic Regression: .703
- KNN Classifier: .726
- Gaussian Naive Bayes: .28
- Bernoullie Naive Bayes: .63
- Support Vector Machine: .736
- Random Forest: .743
- XGBOOST Classifer: .743
## Optimization
I then optimized the Random Forest & XGBOOST Classifer because of their higher scores. Using gridsearch CV over several iterations I determined the optimized
paramerter were max_depth=6, n_estimators = 100, learning_rate=.75. Then I compared our vanilla model to the optimized model using accuracy.
Vanilla XG : .743918725731018
Optimized XG : .7469791558316403

For Random Forest I found, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=820, max_depth = 60.
Vanilla RF: .7439862223535283
Optimized RF: .7522222358389772

## Feature Selection
Finally, I construct a pipeline of my process using the optimized Random Forest model. Then I test various combinations of features against what I buil the model on.
## Final Model
After determing the best featrues to include my final model received a .7594167679222357 accuracy score on an unseen validation set.

# Analysis
From the final model I derived that tthe most important features are:
  1. GPS Height
  2. Water Point Type
  3. Extraction Type
  4. Population
  5. Amount of water available to waterpoint
# Conclusions
- Water points in lower altitude regions have more dysfunctional water pumps
- When the waterpoint source is unknown there are more dysfuncttional water pumps.
- Where the extraction method is unkown there are more dysfunctional water pumps

# Author
Please reach out to me at:
https://www.linkedin.com/in/vincent-404/
or email me at:
jvincentwelsh99@hotmail.com



