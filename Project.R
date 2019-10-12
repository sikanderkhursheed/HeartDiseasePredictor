#Initializing Libraries
library(ggplot2)
library(ISLR)
library(caret)
library(pROC)
library(randomForest)
library (class)
library(e1071)

#Setting Working Directory
setwd("C:\\Users\\HP\\Desktop\\ExpertSystem\\HeartAttackProject")

#Loading data from csv file
data <- read.csv("heart_attack.csv", sep = ",", header = T)

#To remove null values
data <- na.omit(data)

#It shows the structure of dataset
str(data)

#Check dimensions of the dataset
dim(data)

#Converting numeric value of num to factor
data$num<-factor(ifelse(data$num > 0,"Disease","noDisease"))

#Return total count of the factors in the num column
table(data$num)

#Plotting barplot
barplot(table(data$num),
        main="Plot of Disease availability")

#Converting numeric value of sex to factor
data$Sex<-factor(ifelse(data$Sex==0,"female","male"))

table(data$Sex)
table(Sex = data$Sex, disease = data$num )

#Plotting disease based on sex
mosaicplot(data$Sex ~ data$num,
           main="Disease by Gender", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")

boxplot(data$Age ~ data$num,
        main="Fate by Age",
        ylab="Age",xlab="Heart disease")

#plot between age and maximum heart rate 
ggplot(data,aes(x = Age,y = max.heart.rate )) +
  geom_point() +
  geom_smooth()

#Plotting Relationship between Variables
pairs(data)

#Splitting in test and train data
tsize <- floor(0.7 * nrow(data))
dt <- sample(x = nrow(data), size = tsize)

dt.train = data[dt,]
dt.test = data[-dt,]
View(dt.test)
dim(dt.train)
dim(dt.test)

#To store the area under the ROC 
AUC = list()
#To store Accuracy
Accuracy = list()

#Logistic Regression
set.seed(10)
#Model training
logRegModel <- train(num ~ ., data=dt.train, method = 'glm', family = 'binomial')
#Predicting values for test data
logRegPrediction <- predict(logRegModel, dt.test)
#Probability of Desease
logRegPredictionprobDes <- predict(logRegModel, dt.test, type='prob')[1]
#Probability of non desease
logRegPredictionprobNo <- predict(logRegModel, dt.test, type='prob')[2]
#Confusion Matrix
logRegConfMat <- confusionMatrix(logRegPrediction, dt.test[,"num"])

AUC$logReg <- roc(as.numeric(dt.test$num),as.numeric(as.matrix((logRegPredictionprobNo))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']

summary(logRegModel)$coeff

#Random Forest

#Model training
RFModel <- randomForest(num ~ .,
                        data=dt.train,
                        importance=TRUE,
                        ntree=2000)
varImpPlot(RFModel)
#Predicting values for test data
RFPrediction <- predict(RFModel, dt.test)
#Probability of Desease
RFPredictionprobDes = predict(RFModel,dt.test,type="prob")[,1]
#Probability of non desease
RFPredictionprobNo = predict(RFModel,dt.test,type="prob")[, 2]
#Confusion Matrix
RFConfMat <- confusionMatrix(RFPrediction, dt.test[,"num"])

AUC$RF <- roc(as.numeric(dt.test$num),as.numeric(as.matrix((RFPredictionprobNo))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']  

#Boosted tree model
set.seed(10)
objControl <- trainControl(method='cv', number=10,  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
#Train Model
boostModel <- train(num ~ .,data=dt.train, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)

#Predicting values for test data
boostPrediction <- predict(boostModel, dt.test)
#Probability of Desease
boostPredictionprobDes <- predict(boostModel, dt.test, type='prob')[1]
#Probability of Non Desease
boostPredictionprobNo <- predict(boostModel, dt.test, type='prob')[2]
#Confusion Matrix
boostConfMat <- confusionMatrix(boostPrediction, dt.test[,"num"])

#ROC Curve
AUC$boost <- roc(as.numeric(dt.test$num),as.numeric(as.matrix((boostPredictionprobNo))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']  


#GBM
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using the following function
                           summaryFunction = twoClassSummary)
#Train Model
gbmModel <- train(num ~ ., data = dt.train,
                  method = "gbm",
                  trControl = fitControl,
                  verbose = FALSE,
                  tuneGrid = gbmGrid,
                  ## Specify which metric to optimize
                  metric = "ROC")

#Predicting values for test data
gbmPrediction <- predict(gbmModel, dt.test)
#Probability of Desease
gbmPredictionprobDis <- predict(gbmModel, dt.test, type='prob')[1]
#Probability of Non Desease
gbmPredictionprobNo <- predict(gbmModel, dt.test, type='prob')[2]
#Confusion Matrix
gbmConfMat <- confusionMatrix(gbmPrediction, dt.test[,"num"])

#ROC Curve
AUC$gbm <- roc(as.numeric(dt.test$num),as.numeric(as.matrix((gbmPredictionprobNo))))$auc
Accuracy$gbm <- gbmConfMat$overall['Accuracy']

#Support Vector Machine
#Train Model
svmModel <- train(num ~ ., data = dt.train,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
#PRedict values for test data
svmPrediction <- predict(svmModel, dt.test)
#Probability of Desease
svmPredictionprobDis <- predict(svmModel, dt.test, type='prob')[1]
#Probability of Non Desease
svmPredictionprobNo <- predict(svmModel, dt.test, type='prob')[2]
#Confussion Matrix
svmConfMat <- confusionMatrix(svmPrediction, dt.test[,"num"])

#ROC Curve
AUC$svm <- roc(as.numeric(dt.test$num),as.numeric(as.matrix((svmPredictionprobNo))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']


row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
#Showing the accuracy table 
total <- cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 5, ncol = 2,
                           dimnames = list(row.names, col.names))))
