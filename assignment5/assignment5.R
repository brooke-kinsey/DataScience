library(GGally)
library(ggplot2)
library(psych)
library(cluster)
library(dendextend)
library(colorspace)
library(factoextra)
library(class)
library(rpart)
library(rpart.plot)
library(caret)
library(ggfortify)
library(randomForest)
library(dplyr)
library(tidyr)

setwd("C:/Users/brook/Downloads/DataScience/assignment5")

bean.data <- read.csv("Dry_Bean_Dataset.csv")

################################################################################
#1. Exploratory Data Analysis (4000 5% / 6000 3%) - 
#Examine a subset of features in the dataset both separately and in pairs. 
#This may involve cleaning the dataset, for example removing missing values, 
#applying transformations such as log scale and/or taking subsets. 
#You have complete freedom in the selection of features to explore and any 
#necessary filtering. Make sure to use suitable plots to examine variable 
#distributions as well as pairwise relationships between variables. 
#Consider how the presence of outliers affects the plots and consider 
#removing them if they have a significant effect. Explain what you learned 
#about the dataset in 5-6 sentences. 
################################################################################

# Storing variables
Area <- bean.data$Area
Perimeter <- bean.data$Perimeter
MajorAxisLength <- bean.data$MajorAxisLength
MinorAxisLength <- bean.data$MinorAxisLength
AspectRatio <- bean.data$AspectRation
Eccentricity <- bean.data$Eccentricity
ConvexArea <- bean.data$ConvexArea
EquivDiameter <- bean.data$EquivDiameter
Extent <- bean.data$Extent
Solidity <- bean.data$Solidity
Roundness <- bean.data$Roundness
Compactness <- bean.data$Compactness
ShapeFactor1 <- bean.data$ShapeFactor1
ShapeFactor2 <- bean.data$ShapeFactor2
ShapeFactor3 <- bean.data$ShapeFactor3
ShapeFactor4 <- bean.data$ShapeFactor4
Class <- bean.data$Class

# Boxplots
boxplot(Area)
boxplot(Perimeter)
boxplot()

# Simple scatterplots
ggplot(bean.data, aes(x = Area, y = Class)) +
  geom_point() +
  labs(title = "Area vs Class",
       x = "Area",
       y = "Class")

ggplot(bean.data, aes(x = Perimeter, y = Class)) +
  geom_point() +
  labs(title = "Perimeter vs Class",
       x = "Perimeter",
       y = "Class")

ggplot(bean.data, aes(x = MajorAxisLength, y = Class)) +
  geom_point() +
  labs(title = "MajorAxisLength vs Class",
       x = "MajorAxisLength",
       y = "Class")

ggplot(bean.data, aes(x = MinorAxisLength, y = Class)) +
  geom_point() +
  labs(title = "MinorAxisLength vs Class",
       x = "MinorAxisLength",
       y = "Class")

ggplot(bean.data, aes(x = AspectRatio, y = Class)) +
  geom_point() +
  labs(title = "AspectRatio vs Class",
       x = "AspectRatio",
       y = "Class")

#Eccentricity 
ggplot(bean.data, aes(x = Eccentricity, y = Class)) +
  geom_point() +
  labs(title = "Eccentricity vs Class",
       x = "Eccentricity",
       y = "Class")

#ConvexArea
ggplot(bean.data, aes(x = ConvexArea, y = Class)) +
  geom_point() +
  labs(title = "ConvexArea vs Class",
       x = "ConvexArea",
       y = "Class")

#EquivDiameter
ggplot(bean.data, aes(x = EquivDiameter, y = Class)) +
  geom_point() +
  labs(title = "EquivDiameter vs Class",
       x = "EquivDiameter",
       y = "Class")

#Extent
ggplot(bean.data, aes(x = Extent, y = Class)) +
  geom_point() +
  labs(title = "Extent vs Class",
       x = "Extent",
       y = "Class")

#Solidity 
ggplot(bean.data, aes(x = Solidity, y = Class)) +
  geom_point() +
  labs(title = "Solidity vs Class",
       x = "Solidity",
       y = "Class")

#Roundness
ggplot(bean.data, aes(x = Roundness, y = Class)) +
  geom_point() +
  labs(title = "Roundness vs Class",
       x = "Roundness",
       y = "Class")

#Compactness
ggplot(bean.data, aes(x = Compactness, y = Class)) +
  geom_point() +
  labs(title = "Compactness vs Class",
       x = "Compactness",
       y = "Class")

#ShapeFactor1 
ggplot(bean.data, aes(x = ShapeFactor1, y = Class)) +
  geom_point() +
  labs(title = "ShapeFactor1 vs Class",
       x = "ShapeFactor1",
       y = "Class")

#ShapeFactor2 
ggplot(bean.data, aes(x = ShapeFactor2, y = Class)) +
  geom_point() +
  labs(title = "ShapeFactor2 vs Class",
       x = "ShapeFactor2",
       y = "Class")

#ShapeFactor3
ggplot(bean.data, aes(x = ShapeFactor3, y = Class)) +
  geom_point() +
  labs(title = "ShapeFactor3 vs Class",
       x = "ShapeFactor3",
       y = "Class")

#ShapeFactor4
ggplot(bean.data, aes(x = ShapeFactor4, y = Class)) +
  geom_point() +
  labs(title = "ShapeFactor4 vs Class",
       x = "ShapeFactor4",
       y = "Class")


################################################################################
#2. Predictive Modeling (5%) - 
#Decide on a problem to solve by developing predictive models using the 
#data. This could be the same problem for which the dataset was intended or 
#a different problem that you believe can be addressed with this dataset. 
#The solution could involve predicting a continuous variable (regression) or 
#a categorical variable (classification). Consider what you learned about 
#the variables during exploratory analysis to decide which features should 
#be used as inputs to the models. The response variable (output of the model) is 
#usually clear and is dictated by the problem being addressed. 
#Train and evaluate two models and compare their results. 
#The two models should utilize different algorithms (e.g. kNN and Random Forest) 
#but the same set of input variables. Classification models should be evaluated 
#and compared using a confusion matrix and Precision/Recall measurements. 
#Regression models should be evaluated using Mean Squared Error. 
#Explain how models performed and comment on the suitability of the dataset to 
#solve the problem you chose in 5-6 sentences.
################################################################################

#######################################################################
# Classifier (KNN Model)
#######################################################################
set.seed(123)
n <- nrow(bean.data)
train.idx <- sample(1:n, size = 0.7*n)

train.data <- bean.data[train.idx, ]
test.data  <- bean.data[-train.idx, ]

train.y <- train.data$Class
test.y  <- test.data$Class

# Select subset of predictors
predictors <- c("Area", "Perimeter", "MajorAxisLength", "MinorAxisLength") 

train.x <- train.data[, predictors]
test.x  <- test.data[, predictors]

# Find best k value
k.values <- 1:25
accuracies <- numeric(length(k.values))

# Loop over k
for (i in seq_along(k.values)) {
  k <- k.values[i]
  knn.pred <- knn(train = train.x, test = test.x, cl = train.y, k = k)
  accuracies[i] <- mean(knn.pred == test.y)
}

# Find best k
best.k <- k.values[which.max(accuracies)]
best.accuracy <- max(accuracies)

cat("Best k:", best.k, "\n")
cat("Accuracy with best k:", best.accuracy, "\n")

# KNN model
knn.pred <- knn(train = train.x, test = test.x, cl = train.y, k = 3)

# Evaluation
confusion <- table(Predicted = knn.pred, Actual = test.y)
accuracy <- mean(knn.pred == test.y)

# Printing out evaluation
confusion
accuracy

# Precision/Recall
cm_knn <- confusion

# Precision for each class
knn.precision <- diag(cm_knn) / colSums(cm_knn)

# Recall for each class
knn.recall <- diag(cm_knn) / rowSums(cm_knn)

knn.precision
knn.recall


#######################################################################
# Random Forest
#######################################################################
train.data$Class <- as.factor(train.data$Class)
test.data$Class  <- as.factor(test.data$Class)

rf.model <- randomForest(
  Class ~ Area + Perimeter + MajorAxisLength + MinorAxisLength,
  data = train.data,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

# Predictions
rf.pred <- predict(rf.model, test.data)

# Evaluation
rf.confusion <- table(Predicted = rf.pred, Actual = test.y)
rf.accuracy <- mean(rf.pred == test.y)

rf.confusion
rf.accuracy

# Precision/Recall
cm_rf <- rf.confusion

# Precision for each class
rf.precision <- diag(cm_rf) / colSums(cm_rf)

# Recall for each class
rf.recall <- diag(cm_rf) / rowSums(cm_rf)

rf.precision
rf.recall


# Variable importance
importance(rf.model)
varImpPlot(rf.model)
