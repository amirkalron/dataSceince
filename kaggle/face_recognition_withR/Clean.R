
#init
library(doMC)
registerDoMC()

#constannt
data.dir <- '~/git/dataSceince/kaggle/face_r/face_recognition_withR/data'
train.file <- '~/git/dataSceince/kaggle/face_r/face_recognition_withR/data/training.csv'
test.file  <- '~/git/dataSceince/kaggle/face_r/face_recognition_withR/data/test.csv'

#load & clean data
d.train <- read.csv(train.file, stringsAsFactors=F)
im.train  <- d.train$Image
d.train$Image <- NULL
 im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
 }
 
 d.test <- read.csv(test.file, stringsAsFactors=F)
 im.test  <- d.test$Image
 d.test$Image <- NULL
 im.test <- foreach(im = im.test, .combine=rbind) %dopar% {
   as.integer(unlist(strsplit(im, " ")))
 }
 
save(d.train, im.train, d.test, im.test, file='data.Rd')




