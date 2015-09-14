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
data          <- read.table("~/AmazonEC2/king_county/hidden_layer_output.txt", quote="\"", comment.char="")
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

x <- repData[-c(1:2)]
y <- repData[,c(2)]

write.table(x, file="x_hidden_rep.csv", sep = "," , row.names=FALSE)
write.table(y, file="y_hidden_rep.csv", sep = ",", row.names=FALSE)



