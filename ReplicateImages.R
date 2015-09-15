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


# Repeat the same property ID N-times
image_lookup  <- read.csv('/Users/liamf/AmazonEC2/king_county/king_image_table.csv')
data          <- read.table("~/Desktop/hidden_layer_output.txt", quote="\"", comment.char="")
colnames(data)[1] <- 'image_id'
colnames(data)[2] <- 'trueDecile'

data <- merge(data, image_lookup, by="image_id")
#data$predDecile <- NULL
#data$trueDecile <- data$trueDecile + 1  # Lua is 1-indexed 


library(dplyr)

maxImages <- data %>%
             group_by(PropertyID) %>%
             tally() %>%
             filter(n == max(n)) %>%
             select(n) %>%
             unlist(use.names = FALSE)

propertyVec <- unique(data$PropertyID)

repData <- data %>%
             group_by(PropertyID) %>%
             do({addRows(., maxImages)})

x <- repData[-c(1,2,4099)]
y <- repData[,c(2)]

x_train <- x[1:10000,]
y_train <- y[1:10000,]
x_test <- x[10001:20000,]
y_test <- y[10001:20000,]

write.table(x_train, file="/Users/liamf/lstm_data/x_train_mini.csv", sep = "," , row.names=FALSE)
write.table(y_train, file="/Users/liamf/lstm_data/y_train_mini.csv", sep = "," , row.names=FALSE)
write.table(x_test , file="/Users/liamf/lstm_data/x_test_mini.csv" , sep = "," , row.names=FALSE)
write.table(y_test , file="/Users/liamf/lstm_data/y_test_mini.csv" , sep = "," , row.names=FALSE)


