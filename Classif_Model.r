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
# Run the pmm algorithm using mice
pmmMissingdata <- mice(data, m=5, maxit = 8, method = 'pmm')
#summary(pmmMissingdata)

# Fill in the 5 dataframes created
pmmData1 <- complete(pmmMissingdata,1)
pmmData2 <- complete(pmmMissingdata,2)
pmmData3 <- complete(pmmMissingdata,3)
pmmData4 <- complete(pmmMissingdata,4)
pmmData5 <- complete(pmmMissingdata,5)






