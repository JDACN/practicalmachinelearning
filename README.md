# Practical Machine Learning Project
Jayant Dhawale  
September 29, 2017  

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Executive Summary

Based on the [groupware site](http://groupware.les.inf.puc-rio.br/har) , The **Weight Lifting Exercises Dataset** contains data for Six young health participants who were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions. The Results are need to be categorised in Classes like below.


* CLASS A     | Exactly according to the specification
* CLASS B     | Throwing the elbows to the front
* CLASS C     | Lifting the dumbbell only halfway
* CLASS D     | Lowering the dumbbell only halfway
* CLASS E     | Throwing the hips to the front 

This report will describe how the data captured are used to identify the parameters involved in predicting the movement involved based on the above classification, and then used to train and predict.

This training data is divided into two groups, a training data and a validation data (to be used to validate the data), to derived the prediction model by using the training data, to validate the model where an expected out-of-sample error rate of less than 0.5%, or 99.5% accuracy.

This analysis is doneusing Random Forest method to train model as it provided more accuracy compared to other models.

## Load Libraries

In following section we load all needed libraries.

```r
# Load Caret package for Machine Learning
library(caret)
```

## Downloading and Loading Data
Following section download and load data files. The files are downloaded first time and in future local copy is used for analysis. (please note *.csv not added in repository to save space on github account, explicitly  .gitignore is mainted to ignore .csv files. Following code dowload the files from server at first runtime. )


```r
# Prepare URL for Downloading
traingDataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingDataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Download files if not exist !
if (!file.exists("pml_training.csv")) {
      download.file(traingDataURL , "pml_training.csv")
}
if (!file.exists("pml_testing.csv")) {
      download.file(testingDataURL , "pml_testing.csv")
}

# Load Files and set invalid values to NA 
training <- read.csv("pml_training.csv", na.strings = c("NA","#DIV/0!",""))
testing <- read.csv("pml_testing.csv", na.strings = c("NA","#DIV/0!",""))
```

## Make Data Tidy 
Process data to remove invalid predictors as well as reduce the number of predictors by removing columns that have near zero values, NA, or is empty.


```r
# Remove columns with Near Zero Values
subTrain <-  training[, names(training)[!(nzv(training, saveMetrics = T)[, 4])]]
subTest <-  testing[, names(testing)[!(nzv(testing, saveMetrics = T)[, 4])]]

# Remove all columns which contains only NA
subTrain <- subTrain[ , colSums(is.na(subTrain)) == 0]
subTest <-  subTest[ , colSums(is.na(subTest)) == 0]

# Remove columns which may be not useful and may result invalid prediction
## head(subTrain)
## First column looks serial number, also column 5 looks timestamp.
#subTrain <- subTrain[ , c(-1,-5)]

dim(subTrain)
```

```
## [1] 19622    59
```

After intial cleaning of data we have now 59 variables.

## Machine Learning
Following steps are performed to do Machine learning.

### Preprocessing variables
Preprocess the variables to remove the variables with values near zero after variable processing, that means that they have not so much meaning in the predictions. (This is needed to save time in model building and improve the correctness )


```r
# get all columns which are numeric and useful for predictions
v <- which(lapply(subTrain, class) %in% "numeric")

# Preprocess the variables
preObj <-preProcess(subTrain[,v],method=c('knnImpute', 'center', 'scale'))

# Predict best fit variables 
trainLess1 <- predict(preObj, subTrain[,v])
# add the classes for training model.
trainLess1$classe <- subTrain$classe

# Prepare Test Data
testLess1 <-predict(preObj, subTest[,v])

# Remove unwanted columns

nzv <- nearZeroVar(trainLess1,saveMetrics=TRUE)
trainLess1 <- trainLess1[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testLess1,saveMetrics=TRUE)
testLess1 <- testLess1[,nzv$nzv==FALSE]

dim(trainLess1 )
```

```
## [1] 19622    28
```
### Create cross validation sets 
The training set is divided in two parts, one for training and the other for cross validation of our model.

```r
inTrain = createDataPartition(trainLess1$classe, p = 0.6, list=FALSE)
training = trainLess1[inTrain,]
crossValidation = trainLess1[-inTrain,]
```

### Train model
Train model with random forest due to its highly accuracy rate. The model is build on a training set of 27 variables from the initial 160. Cross validation is used as train control method.

```r
modFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

### Check accuracy on training set


```r
trainingPred <- predict(modFit, training)
confusionMatrix(trainingPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
The Training Model looks good with with very high accuracy ! 
### Check accuracy on validation set


```r
cvPred <- predict(modFit, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228    7    0    0    0
##          B    3 1499   11    0    0
##          C    0   12 1347   13    0
##          D    0    0   10 1272    4
##          E    1    0    0    1 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9921          
##                  95% CI : (0.9899, 0.9939)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.99            
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9875   0.9846   0.9891   0.9972
## Specificity            0.9988   0.9978   0.9961   0.9979   0.9997
## Pos Pred Value         0.9969   0.9907   0.9818   0.9891   0.9986
## Neg Pred Value         0.9993   0.9970   0.9968   0.9979   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1911   0.1717   0.1621   0.1833
## Detection Prevalence   0.2849   0.1928   0.1749   0.1639   0.1835
## Balanced Accuracy      0.9985   0.9926   0.9904   0.9935   0.9985
```




```r
cvPred <- predict(modFit, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228    7    0    0    0
##          B    3 1499   11    0    0
##          C    0   12 1347   13    0
##          D    0    0   10 1272    4
##          E    1    0    0    1 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9921          
##                  95% CI : (0.9899, 0.9939)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.99            
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9875   0.9846   0.9891   0.9972
## Specificity            0.9988   0.9978   0.9961   0.9979   0.9997
## Pos Pred Value         0.9969   0.9907   0.9818   0.9891   0.9986
## Neg Pred Value         0.9993   0.9970   0.9968   0.9979   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1911   0.1717   0.1621   0.1833
## Detection Prevalence   0.2849   0.1928   0.1749   0.1639   0.1835
## Balanced Accuracy      0.9985   0.9926   0.9904   0.9935   0.9985
```

The Model looks good with high accuracy nearly 99% with approx 0% sample error on validation dataset too.  Thus we can conclude this model.

# Result

As our model looks good, Now testing the Predictions on the real testing set.


```r
testingPred <- predict(modFit, testLess1)
testingPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
