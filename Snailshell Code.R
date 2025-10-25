# Load Libraries
library(tidyverse)
library(boot)
library(car)
library(caret)
library(randomForest)
library(ggcorrplot)
library(leaps)
library(GGally)

# Load and Prepare Data
data <- read.csv("Snail2.csv")
data$ShellType <- as.factor(data$ShellType)

# A) Exploratory Data Analysis
summary(data)

panel.lm <- function(x, y, ...) {
  points(x, y, ...)
  abline(lm(y ~ x), col = "red", lwd = 1.5)
}

pairs(data[, c("Width", "Length", "AperHt", "AperWdt", "LU", "LipWdt")],
      lower.panel = panel.lm,
      upper.panel = panel.smooth,
      main = "Scatterplot Matrix with Regression Lines")

par(mfrow = c(1, 2))
hist(data$Length, main = "Histogram of Length", xlab = "Length")
boxplot(data$Length, main = "Boxplot of Length")
par(mfrow = c(1, 1))

cor_matrix <- cor(data[, sapply(data, is.numeric)])
ggcorrplot(cor_matrix, lab = TRUE, title = "Correlation Heatmap")

ggplot(data, aes(x = ShellType, y = Length, fill = ShellType)) +
  geom_boxplot(alpha = 0.7) +
  theme_minimal() +
  labs(title = "Length by ShellType", y = "Length")

data %>%
  pivot_longer(cols = c(Width, AperHt, AperWdt, LU, LipWdt)) %>%
  ggplot(aes(x = value, y = Length)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  facet_wrap(~name, scales = "free_x") +
  theme_minimal() +
  labs(y = "Length", title = "Scatterplots: Length vs Predictors")

# B) Response Transformation Check
# Normality Tests for Different Transformations
transforms <- list(
  none = data$Length,
  log = log(data$Length),
  sqrt = sqrt(data$Length),
  square = data$Length^2,
  inverse = 1 / data$Length
)

shapiro_results <- sapply(transforms, function(x) shapiro.test(x)$p.value)
print(shapiro_results)

par(mfrow = c(2, 2))
hist(data$Length, probability = TRUE, main = "Original Length")
curve(dnorm(x, mean = mean(data$Length), sd = sd(data$Length)), add = TRUE, col = "red")
hist(log(data$Length), probability = TRUE, main = "Log(Length)")
curve(dnorm(x, mean = mean(log(data$Length)), sd = sd(log(data$Length))), add = TRUE, col = "blue")
qqnorm(data$Length); qqline(data$Length, col = "red")
qqnorm(log(data$Length)); qqline(log(data$Length), col = "blue")
par(mfrow = c(1, 1))

# T-test for Length by ShellType
t.test(Length ~ ShellType, data = data)

# C) Simple Linear Regression for Each Predictor
data$logLength <- log(data$Length)
predictors <- c("Width", "AperHt", "AperWdt", "LU", "LipWdt")

par(mfrow = c(3, 2))
for (var in predictors) {
  model <- lm(as.formula(paste("logLength ~", var)), data = data)
  plot(data[[var]], data$logLength, main = paste("log(Length) vs", var), xlab = var, ylab = "log(Length)")
  abline(model, col = "red")
  print(summary(model))
}
par(mfrow = c(1, 1))

boxplot(logLength ~ ShellType, data = data, main = "log(Length) by ShellType", ylab = "log(Length)")

# D) Full Multiple Regression
model_all <- lm(logLength ~ Width + AperHt + AperWdt + LU + LipWdt + ShellType, data = data)
summary(model_all)
vif(model_all)

par(mfrow = c(2, 2))
plot(model_all)
par(mfrow = c(1, 1))

# E) Best Subset Selection (Adjusted R²)
regfit.full <- regsubsets(logLength ~ Width + AperHt + AperWdt + LU + LipWdt + ShellType, data = data, nvmax = 6)
reg.sum <- summary(regfit.full)
plot(reg.sum$adjr2, type = "b", xlab = "Number of Predictors", ylab = "Adjusted R²")
best <- which.max(reg.sum$adjr2)
points(best, reg.sum$adjr2[best], col = "red", cex = 2, pch = 19)

# F) Final Interaction Model
model_best <- lm(logLength ~ LU + AperHt + Width + ShellType * AperHt, data = data)
summary(model_best)

par(mfrow = c(2, 2))
plot(model_best)
par(mfrow = c(1, 1))

# G) Confidence and Prediction Intervals
newdata <- data.frame(
  Width = mean(data$Width),
  AperHt = mean(data$AperHt),
  LU = mean(data$LU),
  ShellType = factor(c("Type1", "Type2"), levels = levels(data$ShellType))
)

# Note: Residuals deviate from normality; CI and PI should be interpreted with caution.
pred <- predict(model_best, newdata, interval = "prediction")
conf <- predict(model_best, newdata, interval = "confidence")
print(exp(pred))
print(exp(conf))

# H) Train/Test Split
set.seed(1)
train_index <- createDataPartition(data$Length, p = 0.8, list = FALSE)
trainData <- data[train_index, ]
testData <- data[-train_index, ]
trainData$logLength <- log(trainData$Length)
testData$logLength <- log(testData$Length)

# I) Bagging Model (mtry = 3)
set.seed(2)
bag_model <- randomForest(logLength ~ LU + AperHt + Width, data = trainData, mtry = 3, ntree = 1000)
pred_bag <- exp(predict(bag_model, newdata = testData))
mse_bag <- mean((testData$Length - pred_bag)^2)

# J) Random Forest Model (mtry = 2)
set.seed(3)
rf_model <- randomForest(logLength ~ LU + AperHt + Width, data = trainData, mtry = 2, ntree = 1000)
pred_rf <- exp(predict(rf_model, newdata = testData))
mse_rf <- mean((testData$Length - pred_rf)^2)

# K) Compare Results
mse_df <- data.frame(Method = c("Bagging", "Random Forest"), Test_MSE = c(mse_bag, mse_rf))
ggplot(mse_df, aes(x = Method, y = Test_MSE, fill = Method)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(Test_MSE, 3)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("Bagging" = "salmon", "Random Forest" = "cyan3")) +
  labs(title = "Test MSE: Bagging vs Random Forest", y = "Test MSE") +
  theme_minimal() +
  theme(legend.position = "none")

cat("MSE (Bagging):", round(mse_bag, 4), "\nMSE (Random Forest):", round(mse_rf, 4), "\n")
