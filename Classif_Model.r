install.packages("mice")
library(mice)

data <- read.csv("nci60_binary_class_training_set.csv", header = TRUE, sep = ",")
data$Labels <- as.factor(data$Labels)
is.na(data)

ommitedMissingData <- data

# NAs filled with random value from observed data
randomMissingData <- mice(data, m = 1, maxit = 50, method = 'sample', seed = 500)
View(randomMissingData)
  
# NAs filled with means
meanMissingData <- mice(data, m = 1, maxit = 50, method = 'mean', seed = 500)
View(meanMissingData)

# NAs filled with quadra-ticc terms
quadraticMissingData <- mice(data, m = 1, maxit = 50, method = 'quadratic', seed = 500)
View(quadraticMissingData)
  


