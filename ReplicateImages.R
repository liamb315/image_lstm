library(dplyr)

# Reads in output from CNN, repeats the images so that each property has the same number of images
# then outputs to csv files to be read in by imaage_train.lua

addRows <- function(dat, numRowsNeeded) {
  extraRowsNeeded <- max(0, numRowsNeeded - nrow(dat))
  if(extraRowsNeeded == 0) {
    dat
  } else {
    repsNeeded <- ceiling(extraRowsNeeded/nrow(dat))
    additionalRows <-  rep(1:nrow(dat), repsNeeded)[min(1,extraRowsNeeded):extraRowsNeeded]
    dat[c(1:nrow(dat), additionalRows), ]
  }
}

# Train set
train <- read.csv('/Users/liamf/AmazonEC2/oxford_lstm/image_lstm/raw/train_rnn_prob_dictionary.txt')
colnames(train)[1] <- 'PropertyID'
colnames(train)[2] <- 'ImageID'
colnames(train)[3] <- 'TrueDecile'

maxImages <- 20

propertyVec <- unique(train$PropertyID)
imageVec    <- unique(train$ImageID)

repTrain <- train %>%
  group_by(PropertyID) %>%
  do({addRows(., maxImages)})

x_train <- repTrain[-c(1,2,3)]
y_train <- repTrain[,c(3)]
y_train <- y_train + 1

write.table(x_train, file="/Users/liamf/AmazonEC2/oxford_lstm/image_lstm/data/train/x_train_prob_selldecile.csv", sep = "," , row.names=FALSE)
write.table(y_train, file="/Users/liamf/AmazonEC2/oxford_lstm/image_lstm/data/train/y_train_prob_selldecile.csv", sep = "," , row.names=FALSE)



# Test set
test  <- read.csv('/Users/liamf/lstm_data/test_rnn_dictionary.txt')
colnames(test)[1] <- 'PropertyID'
colnames(test)[2] <- 'ImageID'
colnames(test)[3] <- 'TrueDecile'

maxImages <- 20

propertyVec <- unique(test$PropertyID)
imageVec    <- unique(test$ImageID)

x_test <- repTest[-c(1,2,3)]
y_test <- repTest[,c(3)]
y_test <- y_test + 1

write.table(x_test , file="/Users/liamf/AmazonEC2/oxford_lstm/image_lstm/data/test/x_test_prob_selldecile.csv" , sep = "," , row.names=FALSE)
write.table(y_test , file="/Users/liamf/AmazonEC2/oxford_lstm/image_lstm/data/test/y_test_prob_selldecile.csv" , sep = "," , row.names=FALSE)







