install.packages("randomForest")
install.packages("VIM")
install.packages("mice")
install.packages("BBmisc")

library(randomForest)
library(dplyr)

### READING DATA

# Read the csv
raw_data <- read.csv("nci60_binary_class_training_set.csv", header = TRUE, sep = ",")

# Remove the empty LC.NCI.H23 column
raw_data <- select(raw_data, -LC.NCI.H23)
#View(raw_data)

# Extract non-numeric columns
labels <- raw_data$Labels
ensembl_gene_id = raw_data$ensembl_gene_id

# Make a dataframe with only the numaric data
raw_numeric_data <- raw_data[,6:64]
#View(raw_numeric_data)

### MULTIPLE IMPUTTING METHODS

# NAs ommited
omit_na_data <- na.omit(raw_numeric_data)
#View(ommitedMissingData)

# NAs filled with 0s
zero_na_data <- raw_numeric_data
zero_na_data[is.na(zero_na_data)] <- 0
#View(zero_na_data)

# NAs filled with medians
median_na_data <- na.roughfix(raw_numeric_data)
#View(meanMissingData)

# NAs filled with means
mean_na_data <- raw_numeric_data

for(i in 1:ncol(mean_na_data)){
  mean_na_data[is.na(mean_na_data[,i]), i] <- mean(mean_na_data[,i], na.rm = TRUE)
}
#View(mean_na_data)


# NAs filled with modes
# Declare a function to calculate mode
Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# Create a new dataframe to hold the values and run the function
mode_na_data <- raw_numeric_data
for(i in 1:ncol(mode_na_data)){
  mode_na_data[is.na(mode_na_data[,i]), i] <- Mode(mode_na_data[,i], na.rm = TRUE)
}
#View(mode_na_data)


# NAs filled with k-nearest
library("VIM")
knearest_na_data <- kNN(raw_numeric_data)


# NAs filled with predictive mean matching
# Run the pmm algorithm using mice
library(mice)
pmm_result <- mice(raw_numeric_data, m=5, maxit = 8, method = 'pmm')
#summary(pmm_result)

# Fill in the 5 dataframes created
pmm_na_data_1 <- complete(pmm_result,1)
pmm_na_data_2 <- complete(pmm_result,2)
pmm_na_data_3 <- complete(pmm_result,3)
pmm_na_data_4 <- complete(pmm_result,4)
pmm_na_data_5 <- complete(pmm_result,5)

### DATA NORMALIZATION

# Log2
log2.omit_na_data     <- log2(omit_na_data)   
log2.zero_na_data     <- log2(zero_na_data)
log2.median_na_data   <- log2(median_na_data)
log2.mean_na_data     <- log2(mean_na_data)
log2.mode_na_data     <- log2(mode_na_data)
log2.knearest_na_data <- log2(knearest_na_data)
log2.pmm_na_data_1    <- log2(pmm_na_data_1)
log2.pmm_na_data_2    <- log2(pmm_na_data_2)
log2.pmm_na_data_3    <- log2(pmm_na_data_3)
log2.pmm_na_data_4    <- log2(pmm_na_data_4)
log2.pmm_na_data_5    <- log2(pmm_na_data_5)

# Center-scale (aka z-score)
library(BBmisc)
zscore.omit_na_data     <- normalize(omit_na_data, method = "standardize")
zscore.zero_na_data     <- normalize(zero_na_data, method = "standardize")
zscore.median_na_data   <- normalize(median_na_data, method = "standardize")
zscore.mean_na_data     <- normalize(mean_na_data, method = "standardize")
zscore.mode_na_data     <- normalize(mode_na_data, method = "standardize")
zscore.knearest_na_data <- normalize(knearest_na_data, method = "standardize")
zscore.pmm_na_data_1    <- normalize(pmm_na_data_1, method = "standardize")
zscore.pmm_na_data_2    <- normalize(pmm_na_data_2, method = "standardize")
zscore.pmm_na_data_3    <- normalize(pmm_na_data_3, method = "standardize")
zscore.pmm_na_data_4    <- normalize(pmm_na_data_4, method = "standardize")
zscore.pmm_na_data_5    <- normalize(pmm_na_data_5, method = "standardize")

# Make svm models
library(e1071)
svm_model.log2.zero_na_data <- svm(labels~.,
                                  data = log2.zero_na_data,
                                  kernel = "radial",
                                  cross = 10,
                                  gamma = 0.2,
                                  cost = 1,
                                  fitted = TRUE,
                                  probability = TRUE,
                                  type = "C-classification")


norm.train.pred <- predict(norm.svm.model,
                           norm.train)

confusionMatrix(norm.train.pred,
                norm.train$Severity)

norm.test.pred <- predict(norm.svm.model,
                          norm.test,)

confusionMatrix(norm.test.pred,
                norm.test$Severity)
