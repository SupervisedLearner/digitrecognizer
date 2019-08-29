library(magick)

filepath <- "output"
output_filename <-"output_keras.csv"
fullfilepath <- file.path(filepath, output_filename)
keras_output <- read.csv(fullfilepath)

output_filename <-"output_xgb.csv"
fullfilepath <- file.path(filepath, output_filename)
xgb_output <- read.csv(fullfilepath)

filepath <- "data"
filename <- "test.csv"
fullfilepath <- file.path(filepath, filename)
testdata_raw <- read.csv(fullfilepath)

filtered_data <- keras_output %>% 
  inner_join(xgb_output, by="ImageId") %>% 
  rename(keras_label = Label.x,
         xgb_label = Label.y) %>% 
  bind_cols(testdata_raw) %>% 
  filter(keras_label != xgb_label)

compare_result <- function(i){
  #i<-i
  #dev.off()
  par(mar=c(0, 0, 0, 0))
  test_row <- matrix(unlist(filtered_data[i,-(1:3)]), 28,28, byrow=TRUE)
  test_row <- t(test_row)[,ncol(test_row):1]
  image(test_row, useRaster=TRUE, axes=FALSE, 
        col=gray.colors(256,start=1,end=0))
  c(keras=filtered_data$keras_label[i], xgb=filtered_data$xgb_label[i])
  #dev.off()
}

compare_result(6)
compare_result(11)
compare_result(14)
compare_result(23)
