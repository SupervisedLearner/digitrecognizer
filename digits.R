library(xgboost)
library(tidyverse)

filepath <- "data"
filename <- "train.csv"
fullfilepath <- file.path(filepath, filename)
traindata_raw <- read.csv(fullfilepath)

filename <- "test.csv"
fullfilepath <- file.path(filepath, filename)
testdata_raw <- read.csv(fullfilepath)

filepath <- "output"
output_filename <-"output_xgb.csv"
fullfilepath <- file.path(filepath, output_filename)

set.seed(12345)

#traindata<-mapply(as.integer, traindata_raw[-1])
#trainoutput<-mapply(as.integer,(traindata_raw[1]))

#more VRAM efficient
traindata<-mapply(as.integer, traindata_raw) 
traindata<-xgb.DMatrix(traindata[,-1], label=traindata[,1]) 

testdata<-mapply(as.integer, testdata_raw)
testdata<-xgb.DMatrix(testdata)






#cross validation
elapsed_timecv<-system.time(cvresults <- xgb.cv(
  params    = list(objective="multi:softmax",
                   num_class=10,
                   eta=0.3,
                   max_depth=6),
                   #tree_meth6d="gpu_hist"), #requires GPU version of xgboost
  data      = traindata, 
  nrounds   = 2000,
  nfold     = 5,
  #label     = trainoutput,
  early_stopping_rounds = 10,
  verbose   = TRUE))

elapsed_timecv

#Initial Training using xgboost
elapsed_time<-system.time(model_xgb <- xgboost(
  data        = traindata, 
  #label       = trainoutput,
  #tree_method = "gpu_hist",
  max_depth   = 6, 
  eta         = 0.3, 
  #nthread     = 4, 
  nrounds     = cvresults$best_iteration, # based on CV 
  num_class   = 10,
  objective   = "multi:softmax", 
  verbose     = 1))

elapsed_time


#Predict on the outputs
#target<-predict(model, testdata)
target_xgb<-predict(model_xgb, testdata)

#Write output
output_xgb<-target_xgb %>% 
  enframe() %>% 
  rename(ImageId=name,
         Label=value)

write.csv(output_xgb, fullfilepath, row.names=FALSE)
xgb.save(model_xgb, "output/digits_xgbmodel")

