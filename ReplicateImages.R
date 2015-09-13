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
data          <- read.table("~/AmazonEC2/oxford_lstm/image/data/output_zestimate_error.txt", quote="\"", comment.char="")
names(data)[1:13]<-c("image_id","trueDecile","prob1","prob2","prob3","prob4","prob5","prob6","prob7","prob8","prob9","prob10","predDecile")

data <- merge(data, image_lookup, by="image_id")
data$predDecile <- NULL
data$trueDecile <- data$trueDecile + 1  # Lua is 1-indexed 


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

x <- repData[,c(3:12)]
y <- repData[,c(2)]

write.table(x, file="x_rep.csv", sep = "," , row.names=FALSE)
write.table(y, file="y_rep.csv", sep = ",", row.names=FALSE)



