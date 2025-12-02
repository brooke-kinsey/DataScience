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


# GARDEN AND POOL vs PRICE
lm3 <- lm(Price ~ Has_Pool + Has_Garden, data = house.data)

summary(lm3)

# Plot graph
ggplot(house.data, aes(x = Has_Pool + Has_Garden, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm")

# Plot the residuals.
aug3 <- augment(lm3)
ggplot(aug3, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")


# NUM_FLOORS vs PRICE
lm4 <- lm(Price ~ Location_Score + , data = house.data)

summary(lm4)

# Plot graph
ggplot(house.data, aes(x = Num_Floors, y = Price)) +
  geom_point() +
  stat_smooth(method = "lm")

# Plot the residuals.
aug4 <- augment(lm4)
ggplot(aug4, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")

#################################################################################
# KNN Models
#################################################################################
# Create price range column, using summary
house.data$price.range <- cut(
  house.data$Price,
  breaks = c(276893, 503080, 665942, 960678),
  labels = c("low", "medium", "high"),
  include.lowest = TRUE
)

# Split data into training and testing
n <- nrow(house.data)
s.train <- sample(1:n, 0.7 * n)
house.train <- house.data[s.train, ]
house.test  <- house.data[-s.train, ]



# Convert features to numeric and remove NAs
numeric_features <- c("Square_Feet", "Num_Bedrooms", "Num_Bathrooms",
                      "Location_Score", "Distance_to_Center", "Num_Floors", "Has_Garden")

# Convert and handle NAs
for(f in numeric_features){
  if(f %in% colnames(house.train)){
    house.train[[f]] <- as.numeric(house.train[[f]])
    house.test[[f]]  <- as.numeric(house.test[[f]])
  }
}

# Remove rows with missing values in features or labels
house.train <- na.omit(house.train[, c(numeric_features, "price.range")])
house.test  <- na.omit(house.test[, c(numeric_features, "price.range")])

# Plotting training set
ggplot(house.train, aes(x = Square_Feet, y = Location_Score, color = price.range)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("low" = "red", 
                                "medium" = "green", 
                                "high" = "blue")) +
  labs(
    title = "House Price Clusters by Square Footage and Number of Bedrooms",
    x = "Square Feet",
    y = "Location to Center",
    color = "Price Range"
  )

# Plotting testing set
ggplot(house.test, aes(x = Square_Feet, y = Distance_to_Center, color = price.range)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("low" = "red", 
                                "medium" = "green", 
                                "high" = "blue")) +
  labs(
    title = "House Price Clusters by Square Footage and Number of Bedrooms",
    x = "Square Feet",
    y = "Location to Center",
    color = "Price Range"
  )



# Plotting training set
ggplot(house.train, aes(x = Square_Feet, y = Num_Bedrooms, color = price.range)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("low" = "red", 
                                "medium" = "green", 
                                "high" = "blue")) +
  labs(
    title = "House Price Clusters by Square Footage and Number of Bedrooms (Training Data)",
    x = "Square Feet",
    y = "Number of Bedrooms",
    color = "Price Range"
  )

# Plotting testing set
ggplot(house.test, aes(x = Square_Feet, y = Num_Bedrooms, color = price.range)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("low" = "red", 
                                "medium" = "green", 
                                "high" = "blue")) +
  labs(
    title = "House Price Clusters by Square Footage and Number of Bedrooms (Test Data)",
    x = "Square Feet",
    y = "Number of Bedrooms",
    color = "Price Range"
  )



# Extract labels after removing NAs
train_labels <- house.train$price.range
test_labels  <- house.test$price.range

# Model 1: Square feet, bedrooms
features1 <- c("Square_Feet", "Num_Bedrooms")
train1 <- house.train[, features1]
test1  <- house.test[, features1]

# Model 2: Location score, distance to center
features2 <- c("Location_Score", "Distance_to_Center")
train2 <- house.train[, features2]
test2  <- house.test[, features2]

# Choose k (sqrt of training set size)
k_value <- round(sqrt(nrow(train1)))

# --- Train KNN models ---
knn_pred1 <- class::knn(train = train1, test = test1, cl = train_labels, k = k_value)
knn_pred2 <- class::knn(train = train2, test = test2, cl = train_labels, k = k_value)

# --- Evaluate models ---
cat("Model 1 confusion table:\n")
print(table(Predicted = knn_pred1, Actual = test_labels))

cat("\nModel 2 confusion table:\n")
print(table(Predicted = knn_pred2, Actual = test_labels))

# --- Accuracy ---
acc1 <- mean(knn_pred1 == test_labels)
acc2 <- mean(knn_pred2 == test_labels)

cat("\nAccuracy of Model 1:", round(acc1, 3), "\n")
cat("Accuracy of Model 2:", round(acc2, 3), "\n")

# --- Optional: Compare models ---
contingency_table <- table(Model1 = knn_pred1, Model2 = knn_pred2)
cat("\nContingency table comparing models:\n")
print(contingency_table)

