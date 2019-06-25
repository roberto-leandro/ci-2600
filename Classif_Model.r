install.packages("randomForest")
install.packages("VIM")
install.packages("mice")
install.packages("BBmisc")
install.packages("caret")

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

# Make a dataframe with only the numeric data
raw_numeric_data <- raw_data[,6:64]

# Getting the sample size
sample_size = floor(0.8 * nrow(raw_numeric_data))

set.seed(420)

train_index = sample(seq_len(nrow(raw_numeric_data)), size = sample_size)

raw_numeric_data = raw_numeric_data[train_index,]
raw_numeric_test_data = raw_numeric_data[-train_index,]




#View(raw_numeric_data)

### MULTIPLE IMPUTTING METHODS

# NAs ommited
# Make a new dataframe and add the labels
omit_na_data <- raw_numeric_data
omit_na_data$Labels <- raw_data$Labels

# Remove NAs
omit_na_data <- na.omit(omit_na_data)

# Save the labels with omitted NAs and remove them from the frame
na_labels <- omit_na_data$Labels
omit_na_data <- select(omit_na_data, -Labels)
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

# KNN returns columns with booleans indicating which columns were imputed, so we 
knearest_na_data <- knearest_na_data[,0:59]

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

# log2.zero_na_data has -inf values, so we put them at zero
is.na(log2.zero_na_data) <- sapply(log2.zero_na_data, is.infinite)
log2.zero_na_data[is.na(log2.zero_na_data)] <- 0

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
library(caret)

# Converts labels to factor for use in confusion matrix
labels <- as.factor(labels)
na_labels <- as.factor(na_labels)

# Omit NAs log2
svm_model.log2.omit_na_data <- svm(na_labels~.,
                                     data = log2.omit_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.log2.omit_na_data.pred <- predict(svm_model.log2.omit_na_data,
                                            log2.omit_na_data)

conf_matrix.log2.omit_na_data <- confusionMatrix(svm_model.log2.omit_na_data.pred,
                                 na_labels)

# Omit NAs z-score
svm_model.zscore.omit_na_data <- svm(na_labels~.,
                                   data = zscore.omit_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.zscore.omit_na_data.pred <- predict(svm_model.zscore.omit_na_data,
                                              zscore.omit_na_data)

conf_matrix.zscore.omit_na_data <- confusionMatrix(svm_model.zscore.omit_na_data.pred,
                                   na_labels)


# Zero NAs log2
svm_model.log2.zero_na_data <- svm(labels~.,
                                   data = log2.zero_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.zero_na_data.pred <- predict(svm_model.log2.zero_na_data,
                                            log2.zero_na_data)

conf_matrix.log2.zero_na_data <- confusionMatrix(svm_model.log2.zero_na_data.pred,
                                  labels)

# Zero NAs zscore
svm_model.zscore.zero_na_data <- svm(labels~.,
                                   data = zscore.zero_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.zscore.zero_na_data.pred <- predict(svm_model.zscore.zero_na_data,
                                            zscore.zero_na_data)

conf_matrix.zscore.zero_na_data <- confusionMatrix(svm_model.zscore.zero_na_data.pred,
                                labels)

# Median NAs log2
svm_model.log2.median_na_data <- svm(labels~.,
                                   data = log2.median_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.median_na_data.pred <- predict(svm_model.log2.median_na_data,
                                            log2.median_na_data)

conf_matrix.log2.median_na_data <- confusionMatrix(svm_model.log2.median_na_data.pred,
                                   labels)

# Median NAs zscore
svm_model.zscore.median_na_data <- svm(labels~.,
                                     data = zscore.median_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.median_na_data.pred <- predict(svm_model.zscore.median_na_data,
                                              zscore.median_na_data)

conf_matrix.zscore.median_na_data <- confusionMatrix(svm_model.zscore.median_na_data.pred,
                                     labels)

# Mean NAs log2
svm_model.log2.mean_na_data <- svm(labels~.,
                                   data = log2.mean_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mean_na_data.pred <- predict(svm_model.log2.mean_na_data,
                                            log2.mean_na_data)

conf_matrix.log2.mean_na_data <- confusionMatrix(svm_model.log2.mean_na_data.pred,
                                 labels)

# Mean NAs zscore
svm_model.zscore.mean_na_data <- svm(labels~.,
                                   data = zscore.mean_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.zscore.mean_na_data.pred <- predict(svm_model.zscore.mean_na_data,
                                            zscore.mean_na_data)

conf_matrix.zscore.mean_na_data <- confusionMatrix(svm_model.zscore.mean_na_data.pred,
                                        labels)

# Mode NAs log2
svm_model.log2.mode_na_data <- svm(labels~.,
                                   data = log2.mode_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mode_na_data.pred <- predict(svm_model.log2.mode_na_data,
                                            log2.mode_na_data)

conf_matrix.log2.mode_na_data <- confusionMatrix(svm_model.log2.mode_na_data.pred,
                                        labels)

# Mode NAs zscore
svm_model.zscore.mode_na_data <- svm(labels~.,
                                   data = zscore.mode_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.zscore.mode_na_data.pred <- predict(svm_model.zscore.mode_na_data,
                                            zscore.mode_na_data)

conf_matrix.zscore.mode_na_data <- confusionMatrix(svm_model.zscore.mode_na_data.pred,
                                   labels)

# K-nearest NAs log2
svm_model.log2.knearest_na_data <- svm(labels~.,
                                   data = log2.knearest_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.knearest_na_data.pred <- predict(svm_model.log2.knearest_na_data,
                                            log2.knearest_na_data)

conf_matrix.log2.knearest_na_data <- confusionMatrix(svm_model.log2.knearest_na_data.pred,
                                     labels)

# K-nearest NAs zscore
svm_model.zscore.knearest_na_data <- svm(labels~.,
                                       data = zscore.knearest_na_data,
                                       kernel = "radial",
                                       cross = 10,
                                       gamma = 0.2,
                                       cost = 1,
                                       fitted = TRUE,
                                       probability = TRUE,
                                       type = "C-classification",
                                       na.action = na.omit)

svm_model.zscore.knearest_na_data.pred <- predict(svm_model.zscore.knearest_na_data,
                                                zscore.knearest_na_data)

conf_matrix.zscore.knearest_na_data <- confusionMatrix(svm_model.zscore.knearest_na_data.pred,
                                       labels)

# PMM NAs log2
# PMM 1
svm_model.log2.pmm_na_data_1 <- svm(labels~.,
                                       data = log2.pmm_na_data_1,
                                       kernel = "radial",
                                       cross = 10,
                                       gamma = 0.2,
                                       cost = 1,
                                       fitted = TRUE,
                                       probability = TRUE,
                                       type = "C-classification",
                                       na.action = na.omit)

svm_model.log2.pmm_na_data_1.pred <- predict(svm_model.log2.pmm_na_data_1,
                                                log2.pmm_na_data_1)

conf_matrix.log2.pmm_na_data_1 <- confusionMatrix(svm_model.log2.pmm_na_data_1.pred,
                                  labels)

## PM 2
svm_model.log2.pmm_na_data_2 <- svm(labels~.,
                                    data = log2.pmm_na_data_2,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_2.pred <- predict(svm_model.log2.pmm_na_data_2,
                                             log2.pmm_na_data_2)

conf_matrix.log2.pmm_na_data_2 <- confusionMatrix(svm_model.log2.pmm_na_data_2.pred,
                                  labels)

## PM 3
svm_model.log2.pmm_na_data_3 <- svm(labels~.,
                                    data = log2.pmm_na_data_3,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_3.pred <- predict(svm_model.log2.pmm_na_data_3,
                                             log2.pmm_na_data_3)

conf_matrix.log2.pmm_na_data_3 <- confusionMatrix(svm_model.log2.pmm_na_data_3.pred,
                                  labels)

## PM 4
svm_model.log2.pmm_na_data_4 <- svm(labels~.,
                                    data = log2.pmm_na_data_4,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_4.pred <- predict(svm_model.log2.pmm_na_data_4,
                                             log2.pmm_na_data_4)

conf_matrix.log2.pmm_na_data_4 <- confusionMatrix(svm_model.log2.pmm_na_data_4.pred,
                                  labels)

## PM 5
svm_model.log2.pmm_na_data_5 <- svm(labels~.,
                                    data = log2.pmm_na_data_5,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_5.pred <- predict(svm_model.log2.pmm_na_data_5,
                                             log2.pmm_na_data_5)

conf_matrix.log2.pmm_na_data_5 <- confusionMatrix(svm_model.log2.pmm_na_data_5.pred,
                                  labels)

# PMM NAs zscore
# PMM 1
svm_model.zscore.pmm_na_data_1 <- svm(labels~.,
                                    data = zscore.pmm_na_data_1,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.zscore.pmm_na_data_1.pred <- predict(svm_model.zscore.pmm_na_data_1,
                                             zscore.pmm_na_data_1)

conf_matrix.zscore.pmm_na_data_1 <- confusionMatrix(svm_model.zscore.pmm_na_data_1.pred,
                                    labels)

## PM 2
svm_model.zscore.pmm_na_data_2 <- svm(labels~.,
                                    data = zscore.pmm_na_data_2,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.zscore.pmm_na_data_2.pred <- predict(svm_model.zscore.pmm_na_data_2,
                                             zscore.pmm_na_data_2)

conf_matrix.zscore.pmm_na_data_2 <- confusionMatrix(svm_model.zscore.pmm_na_data_2.pred,
                                    labels)

## PM 3
svm_model.zscore.pmm_na_data_3 <- svm(labels~.,
                                    data = zscore.pmm_na_data_3,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.zscore.pmm_na_data_3.pred <- predict(svm_model.zscore.pmm_na_data_3,
                                             zscore.pmm_na_data_3)

conf_matrix.zscore.pmm_na_data_3 <- confusionMatrix(svm_model.zscore.pmm_na_data_3.pred,
                                    labels)

## PM 4
svm_model.zscore.pmm_na_data_4 <- svm(labels~.,
                                    data = zscore.pmm_na_data_4,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.zscore.pmm_na_data_4.pred <- predict(svm_model.zscore.pmm_na_data_4,
                                             zscore.pmm_na_data_4)

conf_matrix.zscore.pmm_na_data_4 <- confusionMatrix(svm_model.zscore.pmm_na_data_4.pred,
                                    labels)

## PM 5
svm_model.zscore.pmm_na_data_5 <- svm(labels~.,
                                    data = zscore.pmm_na_data_5,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.zscore.pmm_na_data_5.pred <- predict(svm_model.zscore.pmm_na_data_5,
                                             zscore.pmm_na_data_5)

conf_matrix.zscore.pmm_na_data_5 <- confusionMatrix(svm_model.zscore.pmm_na_data_5.pred,
                                    labels)
