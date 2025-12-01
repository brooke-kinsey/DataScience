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
Sqft <- house.data$Square_Feet
Num_Baths <- house.data$Num_Bathrooms
Num_Beds <- house.data$Num_Bedrooms
Location_Score <- house.data$Location_Score
Distance_to_Center <- house.data$Distance_to_Center
Year_Built <- house.data$Year_Built


summary(Sqft)
summary(Num_Baths)
summary(Num_Beds)
summary(Location_Score)
summary(Distance_to_Center)
summary(Year_Built)


hist(Square_Feet)
hist(Num_Bathrooms)
hist(Num_Bedrooms)
hist(Location_Score)
hist(Distance_to_Center)

ggplot(house.data, aes(x = , y = ECO)) +
  geom_point() +
  stat_smooth(method = "lm")

