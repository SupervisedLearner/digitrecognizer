library(xgboost)
library(caret)
library(e1071)
library(tidyverse)

filepath <- "data//"
filename <- "train.csv"
fullfilepath <- paste0(filepath, filename)
traindata_raw <- read.csv(fullfilepath)

filename <- "test.csv"
fullfilepath <- paste0(filepath, filename)
testdata_raw <- read.csv(fullfilepath)

filepath <- "//"
output_filename <-"output.csv"
fullfilepath <- paste0(filepath, output_filename)

set.seed(12345)

traindata<-mapply(as.numeric, traindata_raw[-1])
trainoutput<-mapply(as.numeric,(traindata_raw[1]))
testdata<-mapply(as.numeric, testdata_raw)


#cross validation
elapsed_time<-system.time(cvresults <- xgb.cv(
  params    = list(objective="multi:softmax",
                   num_class=10,
                   eta=0.3,
                   max_depth=6),
  data      = traindata, 
  nrounds   = 500,
  nfold     = 5,
  label     = trainoutput,
  early_stopping_rounds = 10,
  verbose   = TRUE))

elapsed_time

#Initial Training using xgboost
elapsed_time<-system.time(model <- xgboost(
  data      = traindata, 
  label     = trainoutput,
  max_depth = 6, 
  eta       = 0.3, 
  nthread   = 4, 
  nrounds   = 134, # based on CV 
  num_class = 10,
  objective = "multi:softmax", 
  verbose   = 1))

elapsed_time


#Predict on the outputs
#target<-predict(model, testdata)
target<-predict(model, testdata)

#Write output
#output<-bind_cols(testdata_raw["id"], as.data.frame(target))
output<-target %>% 
  as_data_frame() %>% 
  rename(Label=value) %>% 
  mutate(ImageId=1:NROW(target)) %>% 
  select(ImageId, Label)

write.csv(output, fullfilepath, row.names=FALSE)

