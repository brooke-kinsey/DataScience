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
AspectRatio <- bean.data$AspectRatio
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