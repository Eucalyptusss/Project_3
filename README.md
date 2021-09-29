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
I cross validated various models using the train data. I looked at accuracy, precision, and recall. I focused on precision of non functional pump prediction because that is the most relvant metric to meet the business problem. 
- Logistic Regression:
  - Baseline Accuracy: .7677531647222683
  - Baseline (Non Functional) Precision: .799
  - Baseline (Non Functional) Recall: 0.657
  - Baseline (Non Functional) F1: .721
- KNN Classifier: Worse on all metrics
- Gaussian Naive Bayes: Worse on all metrics
- Bernoullie Naive Bayes: Worse on all metrics
- Support Vector Machine: 
  - SVM Accuracy: .7920831807342286
  - SVM (Non Functional) Precision: .799
  - SVM (Non Functional) Recall: .838
  - SVM (Non Functional) F1: .748 
- Random Forest:
  - RF Accuracy: .8046863531014881
  - RF (Non Functional) Precision: .815
  - RF (Non Functional) Recall: .745
  - RF (Non Functional) F1: .779
- XGBOOST Classifer: 
  XG Accuracy: .8032036664262344
  XG (Non Functional) Precision: .828
  XG (Non Functional) Recall: .705
  XG (Non Functional) F1: .705
## Optimization
I then optimized the Random Forest & XGBOOST Classifer because of their higher scores. Using gridsearch CV over several iterations (scoring=precision_macro) For the Gradient Boosed model I determined the optimized parameters are learning_rate =.1, max_depth = 15, n_estimators = 100
- Vanilla XG Accuracy: .8032036664262344
- Vanilla XG (Non Functional) Precision: .828
- Vanilla XG (Non Functional) Recall: .705
- Vanilla XG (Non Functional) F1: .705


- Optimized XG Accuracy: .8121898028107071
- Optimized XG (Non Functional) Precision: .853
- Optimized XG (Non Functional) Recall: .724
- Optimized XG (Non Functional) F1: .779

For Random Forest I found, max_depth = 16, max_features = sqrt, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 100
- Vanilla RF Accuracy: .8046863531014881
- Vanilla RF (Non Functional) Precision: .815
- Vanilla RF (Non Functional) Recall: .745
- Vanilla RF (Non Functional) F1: .779

- Optimized RF Accuracy: .7949588203855564
- Optimized RF (Non Functional) Precision: .864
- Optimized (Non Functional) Recall: .646
- Optimized RF (Non Functional) F1: .739

## Feature Selection
Finally, I construct a pipeline of my process using the optimized Random Forest model. Then I test various combinations of features against what I buil the model on.
## Final Model
- Final Model Accuracy: .812
- Final Model (Non Functional) Precision: .843
- Final Model (Non Functional) Recall: .724
- Final Model (Non Functional) F1: .779

# Analysis
From the final model I derived that tthe most important features are:
  1. Quantity
  2. Water Point Type
  3. Region
  4. LGA
  5. Extraction Type
# Conclusions
-All future water pumps build need a watersource that will not run out.
    -The following rural wards need aid in developing and recording waterpoint sources as they are top contributors to non functional water pumps.
        - Marangu Mashariki
        - Namajani
        - Pangani Mashariki
    -All future water pumps need to be built below 668m altitude when possible.

# Author

Please reach out to me at:
https://www.linkedin.com/in/vincent-404/
or email me at:
jvincentwelsh99@hotmail.com



