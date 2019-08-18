library(keras)
library(tidyverse)

filepath <- "data"
filename <- "train.csv"
fullfilepath <- file.path(filepath, filename)
traindata_raw <- read.csv(fullfilepath)

filename <- "test.csv"
fullfilepath <- file.path(filepath, filename)
testdata_raw <- read.csv(fullfilepath)

filepath <- "output"
output_filename <-"output_keras.csv"
fullfilepath <- file.path(filepath, output_filename)

set.seed(12345)

x_train <- traindata_raw[,-1] / 255
x_train <- as.matrix(x_train)
x_test  <- as.matrix(testdata_raw/255)

y_train <- traindata_raw$label
y_train <- to_categorical(y_train,10)

#define the model and th layers
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu',
                      input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation ='relu') %>% 
  layer_dense(units =10, activation = 'softmax')

#compile model, define loss and optimizer
model %>% 
  compile(
      loss = 'categorical_crossentropy', 
      optimizer = optimizer_rmsprop(),
      metrics = c('accuracy')
  )

#train/fit the model
model %>%
  fit(
    x_train, y_train,
    epochs = 30, 
    batch_size = 18, 
    validation_split = 0.2
  )
  


#run the network on the evaluation set
target <- model %>% predict_classes(x_test)

#Write output
#output<-bind_cols(testdata_raw["id"], as.data.frame(target))
output<-target %>% 
  enframe() %>% 
  rename(ImageId=name,
         Label=value)

write.csv(output, fullfilepath, row.names=FALSE)
