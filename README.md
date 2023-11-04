# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Deeksha P
RegisterNumber:  212222040031
*/
```
```
*/
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(1000000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
1. Result Output

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/a3784b8f-3a7b-4406-9da2-ce927fa855cc)

2. data.head()

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/5a34ffe4-682c-4e36-ad93-c0f9f9b62d3e)

3. data.info()

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/8138416b-5f42-4f81-9df7-f4c900a9e93e)

4. data.isnull().sum()

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/925c403a-fde6-4a9b-845f-e6f2537a4fd6)

5. Y_prediction value

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/dc615ddf-6441-4e49-9e67-acd581dd0df4)

6. Accuracy value

   ![image](https://github.com/Deeksha78/Implementation-of-SVM-For-Spam-Mail-Detection/assets/128116204/326c64f5-4792-4703-bdc2-d02c5fd83de7)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
