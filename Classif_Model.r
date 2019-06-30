install.packages("randomForest")
install.packages("VIM")
install.packages("mice")
install.packages("BBmisc")
install.packages("caret")

library(randomForest)
library(dplyr)

###----------------------------- DATA INPUT -----------------------------###

# Read the csv
raw_data <- read.csv("nci60_binary_class_training_set.csv", header = TRUE, sep = ",")

# Remove the empty LC.NCI.H23 column
raw_data <- select(raw_data, -LC.NCI.H23)
#View(raw_data)

# Make a dataframe with only the numeric data
raw_numeric_data <- raw_data[,6:64]

#View(raw_numeric_data)

###----------------------------- IMPUTTING METHODS -----------------------------###

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

###----------------------------- DATA NORMALIZATION -----------------------------###

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

###----------------------------- TRAIN&TEST DATA PARTITIONING -----------------------------###

# Add labels to one of the omit NA datasets so they also get partitioned (remember, omit NA has a different size)
log2.omit_na_data$Labels <- na_labels

# Add labels to one of the datasets so they also get partitioned
log2.zero_na_data$Labels <- raw_data$Labels

# Partition each resulting dataset, to have testing and training data
# Make 2 partitions - 80% for training and 20% for testing
sample_size = floor(0.8 * nrow(raw_data))
na.sample_size = floor(0.8 * nrow(log2.omit_na_data))
set.seed(420) # set seed so results are reproducible

train_index = sample(seq_len(nrow(log2.omit_na_data)), size = na.sample_size)
log2.train.omit_na_data        = log2.omit_na_data    [train_index,]
log2.test.omit_na_data         = log2.omit_na_data    [-train_index,]

train_index = sample(seq_len(nrow(log2.zero_na_data)), size = sample_size)
log2.train.zero_na_data        = log2.zero_na_data    [train_index,]    
log2.test.zero_na_data         = log2.zero_na_data    [-train_index,]                 

train_index = sample(seq_len(nrow(log2.median_na_data)), size = sample_size)
log2.train.median_na_data      = log2.median_na_data  [train_index,]                    
log2.test.median_na_data       = log2.median_na_data  [-train_index,] 

train_index = sample(seq_len(nrow(log2.mean_na_data)), size = sample_size)
log2.train.mean_na_data        = log2.mean_na_data    [train_index,]                    
log2.test.mean_na_data         = log2.mean_na_data    [-train_index,]

train_index = sample(seq_len(nrow(log2.mode_na_data)), size = sample_size)
log2.train.mode_na_data        = log2.mode_na_data    [train_index,]                    
log2.test.mode_na_data         = log2.mode_na_data    [-train_index,]                 

train_index = sample(seq_len(nrow(log2.knearest_na_data)), size = sample_size)
log2.train.knearest_na_data    = log2.knearest_na_data[train_index,]                    
log2.test.knearest_na_data     = log2.knearest_na_data[-train_index,] 

train_index = sample(seq_len(nrow(log2.pmm_na_data_1)), size = sample_size)
log2.train.pmm_na_data_1       = log2.pmm_na_data_1   [train_index,]                    
log2.test.pmm_na_data_1        = log2.pmm_na_data_1   [-train_index,]

train_index = sample(seq_len(nrow(log2.pmm_na_data_2)), size = sample_size)
log2.train.pmm_na_data_2       = log2.pmm_na_data_2   [train_index,]                    
log2.test.pmm_na_data_2        = log2.pmm_na_data_2   [-train_index,]  

train_index = sample(seq_len(nrow(log2.pmm_na_data_3)), size = sample_size)
log2.train.pmm_na_data_3       = log2.pmm_na_data_3   [train_index,]                    
log2.test.pmm_na_data_3        = log2.pmm_na_data_3   [-train_index,]  

train_index = sample(seq_len(nrow(log2.pmm_na_data_4)), size = sample_size)
log2.train.pmm_na_data_4       = log2.pmm_na_data_4   [train_index,]    
log2.test.pmm_na_data_4        = log2.pmm_na_data_4   [-train_index,]

train_index = sample(seq_len(nrow(log2.pmm_na_data_5)), size = sample_size)
log2.train.pmm_na_data_5       = log2.pmm_na_data_5   [train_index,]    
log2.test.pmm_na_data_5        = log2.pmm_na_data_5   [-train_index,]

train_index = sample(seq_len(nrow(zscore.omit_na_data)), size = na.sample_size)
zscore.train.omit_na_data      = zscore.omit_na_data    [train_index,]
zscore.test.omit_na_data       = zscore.omit_na_data    [-train_index,]

train_index = sample(seq_len(nrow(zscore.zero_na_data)), size = sample_size)
zscore.train.zero_na_data      = zscore.zero_na_data    [train_index,]
zscore.test.zero_na_data       = zscore.zero_na_data    [-train_index,]

train_index = sample(seq_len(nrow(zscore.median_na_data)), size = sample_size)
zscore.train.median_na_data    = zscore.median_na_data  [train_index,]
zscore.test.median_na_data     = zscore.median_na_data  [-train_index,]

train_index = sample(seq_len(nrow(zscore.mean_na_data)), size = sample_size)
zscore.train.mean_na_data      = zscore.mean_na_data    [train_index,]
zscore.test.mean_na_data       = zscore.mean_na_data    [-train_index,]

train_index = sample(seq_len(nrow(zscore.mode_na_data)), size = sample_size)
zscore.train.mode_na_data      = zscore.mode_na_data    [train_index,]
zscore.test.mode_na_data       = zscore.mode_na_data    [-train_index,]

train_index = sample(seq_len(nrow(zscore.knearest_na_data)), size = sample_size)
zscore.train.knearest_na_data  = zscore.knearest_na_data[train_index,]
zscore.test.knearest_na_data   = zscore.knearest_na_data[-train_index,]

train_index = sample(seq_len(nrow(zscore.pmm_na_data_1)), size = sample_size)
zscore.train.pmm_na_data_1     = zscore.pmm_na_data_1   [train_index,]
zscore.test.pmm_na_data_1      = zscore.pmm_na_data_1   [-train_index,]

train_index = sample(seq_len(nrow(zscore.pmm_na_data_2)), size = sample_size)
zscore.train.pmm_na_data_2     = zscore.pmm_na_data_2   [train_index,]
zscore.test.pmm_na_data_2      = zscore.pmm_na_data_2   [-train_index,]

train_index = sample(seq_len(nrow(zscore.pmm_na_data_3)), size = sample_size)
zscore.train.pmm_na_data_3     = zscore.pmm_na_data_3   [train_index,]
zscore.test.pmm_na_data_3      = zscore.pmm_na_data_3   [-train_index,]

train_index = sample(seq_len(nrow(zscore.pmm_na_data_4)), size = sample_size)
zscore.train.pmm_na_data_4     = zscore.pmm_na_data_4   [train_index,]
zscore.test.pmm_na_data_4      = zscore.pmm_na_data_4   [-train_index,]

train_index = sample(seq_len(nrow(zscore.pmm_na_data_5)), size = sample_size)
zscore.train.pmm_na_data_5     = zscore.pmm_na_data_5   [train_index,]
zscore.test.pmm_na_data_5      = zscore.pmm_na_data_5   [-train_index,]

# Recover all the labels and remove them from their respective datasets
train.labels = log2.train.zero_na_data$Labels
test.labels = log2.test.zero_na_data$Labels
na.train.labels = log2.train.omit_na_data$Labels
na.test.labels = log2.test.omit_na_data$Labels

log2.train.zero_na_data <- select(log2.train.zero_na_data, -Labels)
log2.test.zero_na_data <- select(log2.test.zero_na_data, -Labels)
log2.train.omit_na_data <- select(log2.train.omit_na_data, -Labels)
log2.test.omit_na_data <- select(log2.test.omit_na_data, -Labels)

# Make all the labels factors
train.labels <- as.factor(train.labels)
test.labels <- as.factor(test.labels)
na.train.labels <- as.factor(na.train.labels)
na.test.labels <- as.factor(na.test.labels)



###----------------------------- SVM MODEL TRAINING -----------------------------###

# Make svm models
library(e1071)
library(caret)


# Omit NAs log2
svm_model.log2.omit_na_data <- svm(na.train.labels~.,
                                   data = log2.train.omit_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.omit_na_data.pred <- predict(svm_model.log2.omit_na_data,
                                            log2.train.omit_na_data)

conf_matrix.log2.omit_na_data <- confusionMatrix(svm_model.log2.omit_na_data.pred,
                                                 na.train.labels)

# Omit NAs z-score
svm_model.zscore.omit_na_data <- svm(na.train.labels~.,
                                     data = zscore.train.omit_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.omit_na_data.pred <- predict(svm_model.zscore.omit_na_data,
                                              zscore.train.omit_na_data)

conf_matrix.zscore.omit_na_data <- confusionMatrix(svm_model.zscore.omit_na_data.pred,
                                                   na.train.labels)


# Zero NAs log2
svm_model.log2.zero_na_data <- svm(train.labels~.,
                                   data = log2.train.zero_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.zero_na_data.pred <- predict(svm_model.log2.zero_na_data,
                                            log2.train.zero_na_data)

conf_matrix.log2.zero_na_data <- confusionMatrix(svm_model.log2.zero_na_data.pred,
                                                 train.labels)

# Zero NAs zscore
svm_model.zscore.zero_na_data <- svm(train.labels~.,
                                     data = zscore.train.zero_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.zero_na_data.pred <- predict(svm_model.zscore.zero_na_data,
                                              zscore.train.zero_na_data)

conf_matrix.zscore.zero_na_data <- confusionMatrix(svm_model.zscore.zero_na_data.pred,
                                                   train.labels)

# Median NAs log2
svm_model.log2.median_na_data <- svm(train.labels~.,
                                     data = log2.train.median_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.log2.median_na_data.pred <- predict(svm_model.log2.median_na_data,
                                              log2.train.median_na_data)

conf_matrix.log2.median_na_data <- confusionMatrix(svm_model.log2.median_na_data.pred,
                                                   train.labels)

# Median NAs zscore
svm_model.zscore.median_na_data <- svm(train.labels~.,
                                       data = zscore.train.median_na_data,
                                       kernel = "radial",
                                       cross = 10,
                                       gamma = 0.2,
                                       cost = 1,
                                       fitted = TRUE,
                                       probability = TRUE,
                                       type = "C-classification",
                                       na.action = na.omit)

svm_model.zscore.median_na_data.pred <- predict(svm_model.zscore.median_na_data,
                                                zscore.train.median_na_data)

conf_matrix.zscore.median_na_data <- confusionMatrix(svm_model.zscore.median_na_data.pred,
                                                     train.labels)

# Mean NAs log2
svm_model.log2.mean_na_data <- svm(train.labels~.,
                                   data = log2.train.mean_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mean_na_data.pred <- predict(svm_model.log2.mean_na_data,
                                            log2.train.mean_na_data)

conf_matrix.log2.mean_na_data <- confusionMatrix(svm_model.log2.mean_na_data.pred,
                                                 train.labels)

# Mean NAs zscore
svm_model.zscore.mean_na_data <- svm(train.labels~.,
                                     data = zscore.train.mean_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.mean_na_data.pred <- predict(svm_model.zscore.mean_na_data,
                                              zscore.train.mean_na_data)

conf_matrix.zscore.mean_na_data <- confusionMatrix(svm_model.zscore.mean_na_data.pred,
                                                   train.labels)

# Mode NAs log2
svm_model.log2.mode_na_data <- svm(train.labels~.,
                                   data = log2.train.mode_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mode_na_data.pred <- predict(svm_model.log2.mode_na_data,
                                            log2.train.mode_na_data)

conf_matrix.log2.mode_na_data <- confusionMatrix(svm_model.log2.mode_na_data.pred,
                                                 train.labels)

# Mode NAs zscore
svm_model.zscore.mode_na_data <- svm(train.labels~.,
                                     data = zscore.train.mode_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.mode_na_data.pred <- predict(svm_model.zscore.mode_na_data,
                                              zscore.train.mode_na_data)

conf_matrix.zscore.mode_na_data <- confusionMatrix(svm_model.zscore.mode_na_data.pred,
                                                   train.labels)

# K-nearest NAs log2
svm_model.log2.knearest_na_data <- svm(train.labels~.,
                                       data = log2.train.knearest_na_data,
                                       kernel = "radial",
                                       cross = 10,
                                       gamma = 0.2,
                                       cost = 1,
                                       fitted = TRUE,
                                       probability = TRUE,
                                       type = "C-classification",
                                       na.action = na.omit)

svm_model.log2.knearest_na_data.pred <- predict(svm_model.log2.knearest_na_data,
                                                log2.train.knearest_na_data)

conf_matrix.log2.knearest_na_data <- confusionMatrix(svm_model.log2.knearest_na_data.pred,
                                                     train.labels)

# K-nearest NAs zscore
svm_model.zscore.knearest_na_data <- svm(train.labels~.,
                                         data = zscore.train.knearest_na_data,
                                         kernel = "radial",
                                         cross = 10,
                                         gamma = 0.2,
                                         cost = 1,
                                         fitted = TRUE,
                                         probability = TRUE,
                                         type = "C-classification",
                                         na.action = na.omit)

svm_model.zscore.knearest_na_data.pred <- predict(svm_model.zscore.knearest_na_data,
                                                  zscore.train.knearest_na_data)

conf_matrix.zscore.knearest_na_data <- confusionMatrix(svm_model.zscore.knearest_na_data.pred,
                                                       train.labels)

# PMM NAs log2
# PMM 1
svm_model.log2.pmm_na_data_1 <- svm(train.labels~.,
                                    data = log2.train.pmm_na_data_1,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_1.pred <- predict(svm_model.log2.pmm_na_data_1,
                                             log2.train.pmm_na_data_1)

conf_matrix.log2.pmm_na_data_1 <- confusionMatrix(svm_model.log2.pmm_na_data_1.pred,
                                                  train.labels)

## PM 2
svm_model.log2.pmm_na_data_2 <- svm(train.labels~.,
                                    data = log2.train.pmm_na_data_2,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_2.pred <- predict(svm_model.log2.pmm_na_data_2,
                                             log2.train.pmm_na_data_2)

conf_matrix.log2.pmm_na_data_2 <- confusionMatrix(svm_model.log2.pmm_na_data_2.pred,
                                                  train.labels)

## PM 3
svm_model.log2.pmm_na_data_3 <- svm(train.labels~.,
                                    data = log2.train.pmm_na_data_3,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_3.pred <- predict(svm_model.log2.pmm_na_data_3,
                                             log2.train.pmm_na_data_3)

conf_matrix.log2.pmm_na_data_3 <- confusionMatrix(svm_model.log2.pmm_na_data_3.pred,
                                                  train.labels)

## PM 4
svm_model.log2.pmm_na_data_4 <- svm(train.labels~.,
                                    data = log2.train.pmm_na_data_4,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_4.pred <- predict(svm_model.log2.pmm_na_data_4,
                                             log2.train.pmm_na_data_4)

conf_matrix.log2.pmm_na_data_4 <- confusionMatrix(svm_model.log2.pmm_na_data_4.pred,
                                                  train.labels)

## PM 5
svm_model.log2.pmm_na_data_5 <- svm(train.labels~.,
                                    data = log2.train.pmm_na_data_5,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_5.pred <- predict(svm_model.log2.pmm_na_data_5,
                                             log2.train.pmm_na_data_5)

conf_matrix.log2.pmm_na_data_5 <- confusionMatrix(svm_model.log2.pmm_na_data_5.pred,
                                                  train.labels)

# PMM NAs zscore
# PMM 1
svm_model.zscore.pmm_na_data_1 <- svm(train.labels~.,
                                      data = zscore.train.pmm_na_data_1,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_1.pred <- predict(svm_model.zscore.pmm_na_data_1,
                                               zscore.train.pmm_na_data_1)

conf_matrix.zscore.pmm_na_data_1 <- confusionMatrix(svm_model.zscore.pmm_na_data_1.pred,
                                                    train.labels)

## PM 2
svm_model.zscore.pmm_na_data_2 <- svm(train.labels~.,
                                      data = zscore.train.pmm_na_data_2,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_2.pred <- predict(svm_model.zscore.pmm_na_data_2,
                                               zscore.train.pmm_na_data_2)

conf_matrix.zscore.pmm_na_data_2 <- confusionMatrix(svm_model.zscore.pmm_na_data_2.pred,
                                                    train.labels)

## PM 3
svm_model.zscore.pmm_na_data_3 <- svm(train.labels~.,
                                      data = zscore.train.pmm_na_data_3,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_3.pred <- predict(svm_model.zscore.pmm_na_data_3,
                                               zscore.train.pmm_na_data_3)

conf_matrix.zscore.pmm_na_data_3 <- confusionMatrix(svm_model.zscore.pmm_na_data_3.pred,
                                                    train.labels)

## PM 4
svm_model.zscore.pmm_na_data_4 <- svm(train.labels~.,
                                      data = zscore.train.pmm_na_data_4,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_4.pred <- predict(svm_model.zscore.pmm_na_data_4,
                                               zscore.train.pmm_na_data_4)

conf_matrix.zscore.pmm_na_data_4 <- confusionMatrix(svm_model.zscore.pmm_na_data_4.pred,
                                                    train.labels)

## PM 5
svm_model.zscore.pmm_na_data_5 <- svm(train.labels~.,
                                      data = zscore.train.pmm_na_data_5,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_5.pred <- predict(svm_model.zscore.pmm_na_data_5,
                                               zscore.train.pmm_na_data_5)

conf_matrix.zscore.pmm_na_data_5 <- confusionMatrix(svm_model.zscore.pmm_na_data_5.pred,
                                                    train.labels)


###----------------------------- SVM MODEL PERFORMANCE TESTING -----------------------------###

# Omit NAs log2
svm_model.log2.omit_na_data <- svm(na.test.labels~.,
                                   data = log2.test.omit_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.omit_na_data.pred <- predict(svm_model.log2.omit_na_data,
                                            log2.test.omit_na_data)

conf_matrix.log2.omit_na_data <- confusionMatrix(svm_model.log2.omit_na_data.pred,
                                                 na.test.labels)

# Omit NAs z-score
svm_model.zscore.omit_na_data <- svm(na.test.labels~.,
                                     data = zscore.test.omit_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.omit_na_data.pred <- predict(svm_model.zscore.omit_na_data,
                                              zscore.test.omit_na_data)

conf_matrix.zscore.omit_na_data <- confusionMatrix(svm_model.zscore.omit_na_data.pred,
                                                   na.test.labels)


# Zero NAs log2
svm_model.log2.zero_na_data <- svm(test.labels~.,
                                   data = log2.test.zero_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.zero_na_data.pred <- predict(svm_model.log2.zero_na_data,
                                            log2.test.zero_na_data)

conf_matrix.log2.zero_na_data <- confusionMatrix(svm_model.log2.zero_na_data.pred,
                                                 test.labels)

# Zero NAs zscore
svm_model.zscore.zero_na_data <- svm(test.labels~.,
                                     data = zscore.test.zero_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.zero_na_data.pred <- predict(svm_model.zscore.zero_na_data,
                                              zscore.test.zero_na_data)

conf_matrix.zscore.zero_na_data <- confusionMatrix(svm_model.zscore.zero_na_data.pred,
                                                   test.labels)

# Median NAs log2
svm_model.log2.median_na_data <- svm(test.labels~.,
                                     data = log2.test.median_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.log2.median_na_data.pred <- predict(svm_model.log2.median_na_data,
                                              log2.test.median_na_data)

conf_matrix.log2.median_na_data <- confusionMatrix(svm_model.log2.median_na_data.pred,
                                                   test.labels)

# Median NAs zscore
svm_model.zscore.median_na_data <- svm(test.labels~.,
                                       data = zscore.test.median_na_data,
                                       kernel = "radial",
                                       cross = 10,
                                       gamma = 0.2,
                                       cost = 1,
                                       fitted = TRUE,
                                       probability = TRUE,
                                       type = "C-classification",
                                       na.action = na.omit)

svm_model.zscore.median_na_data.pred <- predict(svm_model.zscore.median_na_data,
                                                zscore.test.median_na_data)

conf_matrix.zscore.median_na_data <- confusionMatrix(svm_model.zscore.median_na_data.pred,
                                                     test.labels)

# Mean NAs log2
svm_model.log2.mean_na_data <- svm(test.labels~.,
                                   data = log2.test.mean_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mean_na_data.pred <- predict(svm_model.log2.mean_na_data,
                                            log2.test.mean_na_data)

conf_matrix.log2.mean_na_data <- confusionMatrix(svm_model.log2.mean_na_data.pred,
                                                 test.labels)

# Mean NAs zscore
svm_model.zscore.mean_na_data <- svm(test.labels~.,
                                     data = zscore.test.mean_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.mean_na_data.pred <- predict(svm_model.zscore.mean_na_data,
                                              zscore.test.mean_na_data)

conf_matrix.zscore.mean_na_data <- confusionMatrix(svm_model.zscore.mean_na_data.pred,
                                                   test.labels)

# Mode NAs log2
svm_model.log2.mode_na_data <- svm(test.labels~.,
                                   data = log2.test.mode_na_data,
                                   kernel = "radial",
                                   cross = 10,
                                   gamma = 0.2,
                                   cost = 1,
                                   fitted = TRUE,
                                   probability = TRUE,
                                   type = "C-classification",
                                   na.action = na.omit)

svm_model.log2.mode_na_data.pred <- predict(svm_model.log2.mode_na_data,
                                            log2.test.mode_na_data)

conf_matrix.log2.mode_na_data <- confusionMatrix(svm_model.log2.mode_na_data.pred,
                                                 test.labels)

# Mode NAs zscore
svm_model.zscore.mode_na_data <- svm(test.labels~.,
                                     data = zscore.test.mode_na_data,
                                     kernel = "radial",
                                     cross = 10,
                                     gamma = 0.2,
                                     cost = 1,
                                     fitted = TRUE,
                                     probability = TRUE,
                                     type = "C-classification",
                                     na.action = na.omit)

svm_model.zscore.mode_na_data.pred <- predict(svm_model.zscore.mode_na_data,
                                              zscore.test.mode_na_data)

conf_matrix.zscore.mode_na_data <- confusionMatrix(svm_model.zscore.mode_na_data.pred,
                                                   test.labels)

# K-nearest NAs log2
svm_model.log2.knearest_na_data <- svm(test.labels~.,
                                       data = log2.test.knearest_na_data,
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
                                                     test.labels)

# K-nearest NAs zscore
svm_model.zscore.knearest_na_data <- svm(test.labels~.,
                                         data = zscore.test.knearest_na_data,
                                         kernel = "radial",
                                         cross = 10,
                                         gamma = 0.2,
                                         cost = 1,
                                         fitted = TRUE,
                                         probability = TRUE,
                                         type = "C-classification",
                                         na.action = na.omit)

svm_model.zscore.knearest_na_data.pred <- predict(svm_model.zscore.knearest_na_data,
                                                  zscore.test.knearest_na_data)

conf_matrix.zscore.knearest_na_data <- confusionMatrix(svm_model.zscore.knearest_na_data.pred,
                                                       test.labels)

# PMM NAs log2
# PMM 1
svm_model.log2.pmm_na_data_1 <- svm(test.labels~.,
                                    data = log2.test.pmm_na_data_1,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_1.pred <- predict(svm_model.log2.pmm_na_data_1,
                                             log2.test.pmm_na_data_1)

conf_matrix.log2.pmm_na_data_1 <- confusionMatrix(svm_model.log2.pmm_na_data_1.pred,
                                                  test.labels)

## PM 2
svm_model.log2.pmm_na_data_2 <- svm(test.labels~.,
                                    data = log2.test.pmm_na_data_2,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_2.pred <- predict(svm_model.log2.pmm_na_data_2,
                                             log2.test.pmm_na_data_2)

conf_matrix.log2.pmm_na_data_2 <- confusionMatrix(svm_model.log2.pmm_na_data_2.pred,
                                                  test.labels)

## PM 3
svm_model.log2.pmm_na_data_3 <- svm(test.labels~.,
                                    data = log2.test.pmm_na_data_3,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_3.pred <- predict(svm_model.log2.pmm_na_data_3,
                                             log2.test.pmm_na_data_3)

conf_matrix.log2.pmm_na_data_3 <- confusionMatrix(svm_model.log2.pmm_na_data_3.pred,
                                                  test.labels)

## PM 4
svm_model.log2.pmm_na_data_4 <- svm(test.labels~.,
                                    data = log2.test.pmm_na_data_4,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_4.pred <- predict(svm_model.log2.pmm_na_data_4,
                                             log2.test.pmm_na_data_4)

conf_matrix.log2.pmm_na_data_4 <- confusionMatrix(svm_model.log2.pmm_na_data_4.pred,
                                                  test.labels)

## PM 5
svm_model.log2.pmm_na_data_5 <- svm(test.labels~.,
                                    data = log2.test.pmm_na_data_5,
                                    kernel = "radial",
                                    cross = 10,
                                    gamma = 0.2,
                                    cost = 1,
                                    fitted = TRUE,
                                    probability = TRUE,
                                    type = "C-classification",
                                    na.action = na.omit)

svm_model.log2.pmm_na_data_5.pred <- predict(svm_model.log2.pmm_na_data_5,
                                             log2.test.pmm_na_data_5)

conf_matrix.log2.pmm_na_data_5 <- confusionMatrix(svm_model.log2.pmm_na_data_5.pred,
                                                  test.labels)

# PMM NAs zscore
# PMM 1
svm_model.zscore.pmm_na_data_1 <- svm(test.labels~.,
                                      data = zscore.test.pmm_na_data_1,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_1.pred <- predict(svm_model.zscore.pmm_na_data_1,
                                               zscore.test.pmm_na_data_1)

conf_matrix.zscore.pmm_na_data_1 <- confusionMatrix(svm_model.zscore.pmm_na_data_1.pred,
                                                    test.labels)

## PM 2
svm_model.zscore.pmm_na_data_2 <- svm(test.labels~.,
                                      data = zscore.test.pmm_na_data_2,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_2.pred <- predict(svm_model.zscore.pmm_na_data_2,
                                               zscore.test.pmm_na_data_2)

conf_matrix.zscore.pmm_na_data_2 <- confusionMatrix(svm_model.zscore.pmm_na_data_2.pred,
                                                    test.labels)

## PM 3
svm_model.zscore.pmm_na_data_3 <- svm(test.labels~.,
                                      data = zscore.test.pmm_na_data_3,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_3.pred <- predict(svm_model.zscore.pmm_na_data_3,
                                               zscore.test.pmm_na_data_3)

conf_matrix.zscore.pmm_na_data_3 <- confusionMatrix(svm_model.zscore.pmm_na_data_3.pred,
                                                    test.labels)

## PM 4
svm_model.zscore.pmm_na_data_4 <- svm(test.labels~.,
                                      data = zscore.test.pmm_na_data_4,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_4.pred <- predict(svm_model.zscore.pmm_na_data_4,
                                               zscore.test.pmm_na_data_4)

conf_matrix.zscore.pmm_na_data_4 <- confusionMatrix(svm_model.zscore.pmm_na_data_4.pred,
                                                    test.labels)

## PM 5
svm_model.zscore.pmm_na_data_5 <- svm(test.labels~.,
                                      data = zscore.test.pmm_na_data_5,
                                      kernel = "radial",
                                      cross = 10,
                                      gamma = 0.2,
                                      cost = 1,
                                      fitted = TRUE,
                                      probability = TRUE,
                                      type = "C-classification",
                                      na.action = na.omit)

svm_model.zscore.pmm_na_data_5.pred <- predict(svm_model.zscore.pmm_na_data_5,
                                               zscore.test.pmm_na_data_5)

conf_matrix.zscore.pmm_na_data_5 <- confusionMatrix(svm_model.zscore.pmm_na_data_5.pred,
                                                    test.labels)

results = data.frame(
  
  "Imputing method" = c("Na Omit(Log2)",
                        "Na Omit(Z-Score)",
                        "Na Zero(Log2)",
                        "Na Zero(Z-Score)",
                        "Median(Log2)",
                        "Median(Z-Score)",
                        "Mean(Log2)",
                        "Mean(Z-Score)",
                        "Mode(Log2)",
                        "Mode(Z-Score)",
                        "KNN(Log2)",
                        "KNN(Z-Score)",
                        "PMM(Log2)",
                        "PMM(Z-Score)"),
  
  "Accuracy" = c(0.9932, 
                 0.9911, 
                 0.9929, 
                 0.9912, 
                 0.9877, 
                 0.986, 
                 0.9939, 
                 0.9922, 
                 0.9932,
                 0.9857,
                 0.9925,
                 0.9894,
                 0.9898,
                 0.9867),
  
  "Kappa" = c(0.9599,
              0.9475,
              0.9599,
              0.95,
              0.9294,
              0.9192,
              0.9657,
              0.9559,
              0.9617,
              0.9171,
              0.9578,
              0.9399,
              0.9417,
              0.9234),
  
  "Sensitivity" = c(0.9996,
                    0.9992,
                    0.9992,
                    0.9992,
                    0.9996,
                    0.9992,
                    0.9996,
                    0.9992,
                    0.9996,
                    0.9992,
                    0.9996,
                    0.9992,
                    0.9996,
                    0.9992),
  
  "Specificity" = c(0.9331,
                    0.9155,
                    0.9365,
                    0.9197,
                    0.8829,
                    0.8696,
                    0.9431,
                    0.9298,
                    0.9365,
                    0.8662,
                    0.9298,
                    0.9030,
                    0.9030,
                    0.8763)
)

library(ggplot2)
results <-results[order(-results$Accuracy),]
#counts <- table(results$Accuracy, results$Kappa, results$Sensitivity, results$Specificity)
counts <- data.matrix(results[,2])

hcl_palettes("sequential (multi-hue)", n = 7, plot = TRUE)
hcl_palettes("Oranges")
barplot(counts, 
        main = "Desempeño de los distintos métodos de preprocesamiento", 
        xlab = "Desempeño",
        names.arg = results[,1],
        col = c("#8f4f00", "#a35a00","#b86500" ,"#cc7000" ,"#e07b00" ,"#f58700" ,"#ff8c00" ,"#ff9a1f" ,"#ffa333" ,"#ffac47" ,"#ffb65c" ,"#ffbf70" ,"#ffc885" ,"#ffd199","#ffdaad"),
        xlim = c(0,1),
        legend = rownames(counts),
        beside = TRUE,
        horiz=TRUE)


barplot(counts, horiz=TRUE,  space = 0.4,  yaxp=c(0,25,1), main = "Title", las=1,  
        cex.names=0.8, ylab="y label")










