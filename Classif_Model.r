install.packages("randomForest")
library(randomForest)
library(mice)

completeData <- read.csv("nci60_binary_class_training_set.csv", header = TRUE, sep = ",")

Labels <- completeData$Labels
ensembl_gene_id = completeData$ensembl_gene_id

data2 <- completeData[,6:65]

data <- log2(data2+0.001)

data <- cbind(data, Labels)

data <- cbind(data, ensembl_gene_id)

View(data)

ommitedMissingData <- na.omit(data)
View(ommitedMissingData)
# NAs filled with random value from observed data
randomMissingData <- mice(data, m = 1, maxit = 0, method = 'sample', seed = 500)
View(randomMissingData)
  
# NAs filled with medians
meanMissingData <- na.roughfix(data)
View(meanMissingData)

# NAs filled with quadra-ticc terms
quadraticMissingData <- mice(data, m = 1, maxit = 0, method = 'quadratic', seed = 1)
View(quadraticMissingData)
  


