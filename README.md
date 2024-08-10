# Employee-performance-prediction
Objective

The primary objective of this project is to develop a predictive model to assess employee performance based on various features such as years of experience, education level, job role, and other relevant factors. By accurately predicting employee performance, organizations can make informed decisions related to hiring, training, and resource allocation.

Requirements

Python 3.x
pandas
scikit-learn
matplotlib
numpy

Analysis

The data is supervised and categorical. The predictor variables are ordinal and a few among them are nominal. The target variable 'Performance Rating' is ordinal.

To analyze the data, various data processing techniques like Label Encoding and Standardization is used. Correlation Coeffecient is used to interpret the relationship between variables.

These features include:

Years of Experience: The number of years the employee has worked in their field.

Education Level: The highest level of education attained by the employee.

Job Role: The current role of the employee within the organization.

Performance Scores: Historical performance ratings.

Additional Demographics: Age, gender, department, etc.

For training the data and predicting the target, algorithms used are Logistic Regression, Random Forest, Naive Bayes and XGBoost Classifier.

Summary

The project was done with the purpose of finding out factors which affected the Performance of the employees, training a model which accurately predicts the Performance Rating of the employee, analyzing the data to provide recommendations to improve the performance and gain insights from the analysis. The following steps were carried out:

    Import the data provided, find out the predictor & target variables and look for missing values.

    Analysis of Department wise performance as asked.
    Label Encoding the ordinal columns.

    Calculate correlation coeffecient to find out the relationship between variables and then select the important features for analysis.
    
    Standardizing the data and splitting it into test and train.
    
    Training the data using algorithms like Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, Naive Bayes, K-Nearest Neighbor, XGBoost Classifier and Artificial Neural Network and checking the accuracy to find out which algorithm is the best.
    
    Exporting the model with highest accuracy.



Recommendation
    From the results, we can conclude that the company should provide a better environment as it increases the performance drastically. The company should increase the salary of the employee from time to time and help them maintain a worklife balance. On the other hand, shuffling the manager from time to time will also affect performance.
