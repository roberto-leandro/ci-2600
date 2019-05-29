install.packages("randomForest")
install.packages("VIM")
install.packages("mice")
library(VIM)
library(mice)
library(randomForest)
library(dplyr)

completeData <- read.csv("nci60_binary_class_training_set.csv", header = TRUE, sep = ",")

completeData <- select(completeData, -LC.NCI.H23)
View(completeData)
###
Labels <- completeData$Labels

ensembl_gene_id = completeData$ensembl_gene_id

data2 <- completeData[,6:64]

data <- log2(data2+0.001)

data <- cbind(data, Labels)

data <- cbind(data, ensembl_gene_id)
###

View(data)

# NAs ommited
ommitedMissingData <- na.omit(data)
# View(ommitedMissingData)

# NAs filled with 0s
# TODO

# NAs filled with medians
medianMissingData <- na.roughfix(data)
View(meanMissingData)

# NAs filled with means
meansMissingData <- data

for(i in 1:ncol(meansMissingData)){
  meansMissingData[is.na(meansMissingData[,i]), i] <- mean(meansMissingData[,i], na.rm = TRUE)
}

View(meansMissingData)

# NAs filled with k-nearest
library("VIM")
knearestMissingdata <- kNN(data)


# NAs filled with predictive mean matching
pmmMissingdata <- data
pmmMissingdata <- mice(pmmMissingdata, m=5, maxit = 50, method = 'pmm')


