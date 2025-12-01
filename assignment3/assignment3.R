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

setwd("C:/Users/brook/Downloads/DataScience/assignment3")

house.data <- read.csv("real_estate_dataset.csv")

# EDA
summary(house.data$Square_Feet)
summary(house.data$Num_Bathrooms)
summary(house.data$Num_Bedrooms)
summary(house.data$Location_Score)
summary(house.data$Distance_to_Center)
summary(house.data$Year_built)


hist(house.data$Square_Feet)
hist(house.data$Num_Bathrooms)
hist(house.data$Num_Bedrooms)
hist(house.data$Location_Score)
hist(house.data$Distance_to_Center)

ggplot()

