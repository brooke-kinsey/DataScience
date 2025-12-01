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
library(EnvStats)
library(broom)
library(readr)

setwd("C:/Users/brook/Downloads/DataScience/assignment3")

house.data <- read.csv("real_estate_dataset.csv")

# EDA
Sqft <- house.data$Square_Feet
Num_Baths <- house.data$Num_Bathrooms
Num_Beds <- house.data$Num_Bedrooms
Location_Score <- house.data$Location_Score
Distance_to_Center <- house.data$Distance_to_Center
Year_Built <- house.data$Year_Built
Price <- house.data$Price


# Variable Summaries
summary(Sqft)
summary(Num_Baths)
summary(Num_Beds)
summary(Location_Score)
summary(Distance_to_Center)
summary(Year_Built)
summary(Price)


# Histograms and Barcharts
hist(Sqft)

# THESE ARE BAD
hist(Num_Baths)
hist(Num_Beds)

# BETTER FOR WHOLE NUMBER VALUES
barplot(table(Num_Baths),
        main = "Bar Chart of Num_Baths",
        xlab = "Number of Bathrooms",
        ylab = "Count")

barplot(table(Num_Beds),
        main = "Bar Chart of Num_Baths",
        xlab = "Number of Bathrooms",
        ylab = "Count")


hist(Location_Score)
hist(Distance_to_Center)
hist(Price)

#####################################################################################
# Linear Regression Models
#####################################################################################

# SQFT vs PRICE
sqft.lm <- lm(Price ~ Sqft, data = house.data)

summary(sqft.lm)

# Plot graph
ggplot(house.data, aes(x = Sqft, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm")

# Plot the residuals.
aug1 <- augment(sqft.lm)
ggplot(aug1, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")


# NUM_BEDS + NUM_BATHS vs PRICE
lm2 <- lm(Price ~ Num_Beds + Num_Baths, data = house.data)

summary(lm2)

# Plot graph
ggplot(house.data, aes(x = Num_Beds + Num_Baths, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm")

# Plot the residuals.
aug2 <- augment(lm2)
ggplot(aug2, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")


#  vs PRICE
lm2 <- lm(Price ~ Num_Beds + Num_Baths, data = house.data)

summary(lm2)

# Plot graph
ggplot(house.data, aes(x = Num_Beds + Num_Baths, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm")

# Plot the residuals.
aug2 <- augment(lm2)
ggplot(aug2, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")



#################################################################################
# KNN Model
#################################################################################

n <- nrow(house.data)
s.train <- sample(1:n, 0.7 * n)
house.train <-house.data[s.train,]
house.test <-house.data[-s.train,] 

# Inputs
inputs <- c("Sqft", "Num_Beds", "Num_Baths")

# Training and testing
train1 <- house.train[, inputs]
test1  <- house.test[, inputs]

train_labels <- house.train$region
test_labels  <- house.test$region

# Seeing which value of k is best (from 1-10)
k_values <- 1:10
accuracy <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  pred_i <- knn(train = train1, test = test1, cl = train_labels, k = k_values[i])
  accuracy[i] <- mean(pred_i == test_labels)
}

# Find best k
best_k <- k_values[which.max(accuracy)]

cat("Best k:", best_k, "\n")

knn_pred1 <- knn(train = train1, test = test1, cl = train_labels, k = 1)

# Evaluate model/confusion matrix
print(table(Predicted = knn_pred1, Actual = test_labels))

# Accuracy calculation
acc1 <- mean(knn_pred1 == test_labels)

cat("\nAccuracy of Model 1:", round(acc1, 3), "\n")
