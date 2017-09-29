# Corsera : Practical Machine Learning Project
Jayant Dhawale  
September 29, 2017  

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Executive Summary

Based on the [groupware site](http://groupware.les.inf.puc-rio.br/har) , The **Weight Lifting Exercises Dataset** contains data for Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions. The Results are need to be categorised in Classes like below.


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

# Remove all columns which contains only NA
subTrain <- subTrain[ , colSums(is.na(subTrain)) == 0]

# Remove columns which may be not useful and may result invalid prediction
## head(subTrain)
## First column looks serial number, also column 5 looks timestamp.
subTrain <- subTrain[ , c(-1,-5)]
```

## Machine Learning
Following steps are performed to do Machine learning.

### Create cross validation sets 
The training set is divided in two parts, one for training and the other for cross validation of our model.

```r
inTrain <- createDataPartition(subTrain$classe, p = 0.6, list = FALSE)
subTraining <- subTrain[inTrain,]
subValidation <- subTrain[-inTrain,]
```

### Train model
Train model with random forest due to its highly accuracy rate. The model is build on a training set of 28 variables from the initial 160. Cross validation is used as train control method.


