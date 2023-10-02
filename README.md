# Stroke Prediction Model

## Motivation and Goal

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. 
The silver lining is that strokes are highly preventable and simple lifestyle modifications (such as reducing alcohol and tobacco use; eating healthily and exercising) coupled with early treatment greatly improves its prognosis. It is, however, difficult to identify high-risk patients because of the multifactorial nature of several contributory risk factors such as high blood pressure, high cholesterol, et cetera. This is where machine learning and data mining come to the rescue. The application of such algorithms and interpretation of the patterns can be helpful in saving numerous people's lives by anticipating the condition of the disease in advance.
Given a set of parameters (gender, age, hypertension, average glucose level, if the patient ever has a heart disease, etc.), using different classification algorithms, we want to predict if a patient is susceptible to getting a stroke.
Dataset Description
This Dataset is publicly available in kaggle. The dataset consists of 12 attributes (including target variable) and 5110 observations. The attributes are a combination of ordinal, discrete, and continuous types. Below are the attributes in the dataset:
1.	Id: nominal, discrete 
2.	Gender: nominal, binary
3.	Age: ratio, discrete
4.	Hypertension: nominal, binary
5.	Heart_disease: nominal, binary 
6.	Ever_married: nominal, binary
7.	Work_type: nominal, categorical
8.	Residence_type: nominal, categorical
9.	Avg_glucose_level: ratio, continuous 
10.	bmi: ratio,continuous
11.	Smoking_status: nominal, categorical
12.	stroke: nominal, binary
The last attribute, stroke is the target variable which takes values 1, for stroke and 0, for no stroke.


## Exploratory Data Analysis
Below is the visualization of the target attribute of the dataset i.e., stroke.

 
Distribution Plot of Target Attribute

From the distribution, we can observe that there is a class imbalance problem with the target attribute. 
To gain more insights into the data, we have visualized the distributions of the attributes with respect to the target (i.e., stroke) attribute.
 
 
Categorical Attributes Vs Target Attribute

Insights drawn from the above plot with respect to the Stroke Data
1.	Both the Genders have approximately 5% chance to get a stroke. It is close to the overall stroke percentage and thus can be considered insignificant on its own.
2.	People with a history of Hypertension and Heart Disease have a higher chance of having a stroke. Percentage of Stroke is around 12.5% and 16.5% respectively.
3.	Married/Divorced people have a 6.5% chance of stroke. No wonder why people these days choose to stay single.
4.	Self Employed people have a higher chance compared to Private and Govt Jobs.
5.	Rural and Urban doesn't show much difference and approximately have a 5% chance to get a stroke.
6.	For some reason, people who once used to smoke have a higher chance compared to people who are still smoking.
 
Numerical Attributes Vs Target Attribute
Insights drawn from the above plot with respect to the Stroke Data
1.	Age is a big factor in stroke patients - the older you get the more at risk you are.
2.	People with prediabetic (glucose level between 140-199) and diabetes (glucose level >200) are more prone to having a stroke.
3.	Overweight people (with high BMI) are slightly more likely to have a stroke.
In the final step we checked the correlation of the different features with the target variable and with each other as this would not only give a good estimate of the strength of the features as predictors of stroke but also reveal any collinearity among the features.
 
Correlation Matrix
From the matrix, there are no features with a correlation of more than 0.25 with the target stroke attribute and this shows that the features are poor predictors. However, the features with the highest correlation are age, hypertension, heart_disease and avg_glucose_level.
## Data Preprocessing
We have pre-processed the dataset in four steps:
1. 	Data Imputation
2. 	Encoding categorical attributes
3. 	Class Imbalance
4. 	Feature scaling
### Data Imputation
We have observed that there are missing values in the ‘bmi’ attribute.
 
Target Vs bmi
From the above plot, we can observe that the missing values of bmi (left most column) contribute to a major part of the ‘stroke=yes’ class. Therefore, we couldn’t delete the records containing missing values. We handled these missing values in three ways:
1.	Using Median: In this method, we had replaced the missing values with the median of the bmi attribute. This is a statistical approach of handling the missing values.
2.	Using a Decision Tree to predict BMI for missing rows: We have used a simple decision tree model which based on the age and gender of all other samples gave us a fair prediction for the missing values.
3.	Using a KNN Imputer: We have used a KNN regressor to predict the missing BMI values based on age and gender.  By default, it uses a Euclidean distance metric to impute the missing values.





### Encoding categorical attributes
We have observed that all the categorical attributes in the dataset are nominal. We are converting string values into integer values by assigning a unique integer to each unique value. For example in the attribute gender, we are representing male by 0 and female by 1.
Cleaning some attributes and removing irrelevant attributes
There was one sample that had the gender attribute set to “Other”. We removed that attribute. After going through some visualizations we came to a conclusion to get rid of the following attributes: residence_type, ever_married, work_type.
### Class Imbalance
We have observed that there is an imbalance in the target attribute with only 5% contributing to ‘stroke=yes’. It is not advised to train a classifier on an imbalanced data set as it may be biased towards one class thus achieving high accuracy but have poor sensitivity or specificity. 
We have used a Synthetic Minority Oversampling Technique (SMOTE) to address this problem. SMOTE first selects a minority class instance at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors at random and connecting both to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances.
Below are the visualizations of the target attribute before and after Oversampling.
  
### Feature scaling
We have standardized the attributes in our dataset using a Standard Scaler().Given the distribution of the data, each value in the dataset will have the mean value subtracted, and then divided by the standard deviation of the whole dataset (or feature in the multivariate case). This is part of the pipeline that we build while training models.




## Classifiers
In order to predict the target variable, we decided on using different types of classifiers and then comparing their performance and choosing the one that works best. We chose 3 different classifiers:
1.	Logistic Regression
2.	Random Forests
3.	Support Vector Machines

We chose logistic regression for its simplicity and because our target variable was binary in nature. SVM was a natural choice because of its versatility. We also wanted to use the power of an ensemble method and hence chose random forests.
Our data has a skew of 95% no stroke- 5% stroke. A trivial classifier can classify all incoming entries as negative and achieve an accuracy of 95%. This shows that accuracy as a measure is not a good way to train and evaluate our models. In our project we considered the f1-score to train the models and used the ROC area under the curve to compare them.
In order to train our models, we have taken a conservative split of 70:30 (train:test). After training the model we tested them on the test set. In order to better our results we used gridSearchCV and tuned a set of hyper-parameters. We chose the hyper-parameters through intuition and fed the algorithm a set of acceptable values for them. After getting the most appropriate hyperparameters, we trained the models again, this time with the optimized parameters, and ran them on the test set. 
1.	Random forests:
The first model we tried was the random forest model. We ran this model with the default parameters and got really good results on the training data. But, when we ran it on the test data, we got a very poor f1 score. This gave us an idea about the difficulty of the problem. By using the hyperparameter tuning method mentioned above, we got the following parameters - {'RF__bootstrap': False, 'RF__criterion': 'gini', 'RF__max_features': 3,  'RF__n_estimators': 175}. When we used random forest with the following parameters on the test set, we couldn’t find a huge improvement in the f1-scores. They were practically the same. 
2.	Logistic Regression:
The next model we tried was logistic regression. We ran this model with the default parameters and got the worst results on the training data. But, when we ran it on the test data, we got the best f1 score. By using the hyperparameter tuning method mentioned above, we got the following parameters - {'LR__C': 0.005, 'LR__penalty': 'l1', 'LR__solver': 'liblinear'}. When we used Logistic Regression with the following parameters on the test set we saw that the f1-score was similar but the recall had increased from 0.65 to 0.72. This we believe is a favorable outcome and have used it for further analysis.
3.	SVM:
The next model we tried was support-vector machines. We ran this model with the default parameters and got mediocre results on the training data. We got mediocre results even when we ran it on the test data. By using the hyperparameter tuning method mentioned above, we got the following parameters - {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}. When we used SVM with the following parameters on the test set we didn’t see any improvement.
An observation that we think is interesting is that random forest performed the best and logistic regression the worst in terms of f1-score on training data. When we used test data, random forest performed the worst whereas logistic regression performed the best. This just goes to show that the generalization capability of logistic regression for this dataset is better than the other classifiers.
The different measures for the models on the test set are depicted in the image below.
 
Since logistic regression has the best f1 score among the three, we will expand upon it and show its confusion matrix as well. In the confusion matrix below one obvious and concerning fact is the presence of a high number of false positives. But, we can see that the false positive rate is only around 0.2.
 


In order to compare the different models with each other, we use the ROC area under curve as well.
 
We can notice that logistic regression performs better than the other models at all places and has a better area under the curve. This helped us settle on logistic regression as our final model.



## Model interpretation
### Random Forest
SHAP values (SHapley Additive exPlanations) break down a prediction to show the impact of each feature. It interprets the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value (e.g. zero).


 	 
 	 


The plot above shows the effect of each data point on our predictions. For example, for age, the top-left point reduced the prediction by 0.6. The color shows whether that feature was high or low for that row of the dataset Horizontal location shows whether the effect of that value caused a higher or lower prediction. We can also see how our Random Forest Model is heavily skewed in favour of predicting no-strokes. We can also focus on how the impact of each variable changes, as the variable itself changes. For instance, Age. When this variable increases, the SHAP value also increases - pushing the patient closer to  our 1 condition (stroke). This is also shown with color -  pink/red representing those who suffered a stroke. Here we see a clear cutoff point for when strokes become far more common - after a BMI of around 30 or so.
### Logistic Regression Interpretation
The graphs shown below are generated using ‘lime interpreter’ for logistic regression.
 
This example predicts a stroke probability of 0.72. Next, it tells us the key attributes which influenced its prediction. As we can see, age>60 is thought to influence the probability of stroke to a large extent. Gender being 1 (female) tends to decrease the stroke probability. 
 
This example predicts a stroke probability of 0.19. As we can see, 24<age<=45 is thought to influence the probability of not having a stroke to a large extent making the probability of having a stroke much smaller. Gender being 0 (male) tends to increase the stroke probability, etc. 
Such interpretation is really useful for a practitioner to determine the reasons why a person who’s being screened for stroke gets a particular value for getting stroke. This helps build trust in the model for an individual.
 
The table shown above shows the importance of particular features by evaluating the model globally. It tells us that age and avg_glucose_level have a high impact on stroke.

## Conclusions

### Why do we think logistic regression performs the best?
In our dataset we have some interacting attributes like  avg_glucose_level and bmi and logistic regression is good at handling interacting attributes. The target variable is binary. From the correlation map, we can see that there is very little correlation between attributes. This along with the fact that we got rid of the irrelevant attributes in the pre-processing step are favorable conditions for logistic regression. Lastly, logistic regression is simple in nature and simple models usually have a tendency to generalize better as is evident from this project. 

### How can our model be used?
Our model has a fairly high recall but has low precision. The false-positive rate of our model is low but because of the class imbalance issue, the precision of our model is pedestrian. This means that we will have a fair amount of false positives. We believe that this is a necessary tradeoff in this use case as after getting a false positive the patient can get the opinion of an expert. But a false negative can give the patient a false sense of security and can make the condition worse and can eventually lead to death. 
The end-user needs to be informed about this peculiarity. They should be informed that if they get a positive for stroke they should consult a doctor.

### How can our model interpretation be used?
The LIME analysis that we have done on our logistic regression model can assist a doctor in diagnosing if a patient is susceptible to a stroke or not. 

This was a very challenging dataset and we were able to learn and apply a lot of data mining techniques. We think that we can arrive at better results if we can get hold of a lot more data both in terms of samples and attributes. Sensitive data such as common medical instrument measurements and metrics such as heart rate variability, breathing rate, SpO2 variability can help us explore more about what causes stroke and how we can tackle this widespread lifestyle disease. 






![image](https://github.com/harshithaanuganti/stroke-prediction/assets/74675390/6700303b-c293-4716-8268-a39e1fb2e40b)

