library(caret)
library(ggplot2)

set.seed(32343)
setwd("/home/roger/Escritorio/ML_Course/assignment/")
data <- read.csv('data/pml-training.csv', header=TRUE, sep=",")

#Remove columns with NA
data <- data[sapply(data, function(x) !any(is.na(x)))]

# get indices of data.frame columns with low variance
badCols <- nearZeroVar(data)

# remove those "bad" columns from the data set
data <- data[, -badCols]
data$cvtd_timestamp <- NULL
data$user_name <- NULL

inTrain <- createDataPartition(y = data$classe, p=0.75, list=FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]

# Pre-processing: Box-Cox transformation, centered, scaled and principal component signal extraction
preProc <- preProcess(training[, -57], method=c("BoxCox", "center", "scale", "pca"), thresh= 0.80)
trainPC <- predict(preProc, training[, -57])
preProc

# Plotting some of the relevant principal component interactions
qplot(PC1, PC2, data=trainPC, col=training$classe, xlab="PC1", ylab="PC2")
qplot(PC6, PC2, data=trainPC, col=training$classe, xlab="PC6", ylab="PC2")

# Fit the model with 10-fold cross validation
## check if model exists? If not, refit:
if(file.exists("model/modelRF.rda")) {
        ## load model
        load(file = "model/modelRF.rda")
} else {
        ## (re)fit the model and save it
        train_control <- trainControl(method="cv", number=10)
        modelFit <- train(training$classe ~ ., method="rf", data=trainPC, trControl=train_control, prox=TRUE)
        save(modelFit, file = "model/modelRF.rda")
}

modelFit

pred <- predict(modelFit, trainPC)
confusionMatrix(pred, training$classe)
plot(main="Overall model error", modelFit$finalModel, log="y")


# Apply the same PCA transformation to the testing set
testPC <- predict(preProc, testing[, -57])

# Testing the predictor with the testing set
pred <- predict(modelFit, testPC)
confusionMatrix(testing$classe, pred)

# Plotting correct and incorrect predictions
testPC$predRight <- pred==testing$classe
qplot(PC1, PC2, colour=predRight, data=testPC, main="Newdata Prediction")

# Testing with validation data (20 samples) 
validation <- read.csv('data/pml-testing.csv', header=TRUE, sep=",")

# Remove columns with NAs
validation <- validation[sapply(data, function(x) !any(is.na(x)))]

# get indices of data.frame columns with low variance
badCols <- nearZeroVar(validation)

# remove those "bad" columns from the validation set
validation <- validation[, -badCols]
validation$cvtd_timestamp <- NULL
validation$user_name <- NULL
validationPC <- predict(preProc, validation[, -57])
results <- predict(modelFit, validationPC)

results
