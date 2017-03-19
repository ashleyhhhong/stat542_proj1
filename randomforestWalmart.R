# Convert train.csv into mytrain.csv, mytest.csv, and myxxx1.csv -> myxxx20.csv as 

# NOT required for actual submission, but makes it so we can do testing

# Read in full training data (change the location to where your original train.csv)
# file downloaded from Kaggle is
require(plyr)
train <- read.csv("~/Desktop/walmart/trainWal.csv")
train$Date=as.Date(train$Date)

# mytrain.csv should only go from 2010-02 to 2011-02
mytrain=train[train$Date<as.Date("2011-03-01"),]
mytest=train[train$Date>=as.Date("2011-03-01"),]
myxxx=train[train$Date>=as.Date("2011-03-01"),]

# mytest.csv is everything else, drop the Weekly_Sales and add Weekly_pred(1-3)
mytest=subset(mytest,select=-Weekly_Sales)
mytest$Weekly_Pred1=0
mytest$Weekly_Pred2=0
mytest$Weekly_Pred3=0

# Write these two files
write.csv(mytrain,'train.csv',row.names=FALSE)
write.csv(mytest,'test.csv',row.names=FALSE)

# Make myxxx(1-20).csv files
year=2011
month=3
for(t in 1:20){
  tmp.filename = paste('xxx', t, '.csv', sep='')
  tmp=myxxx[as.numeric(format(myxxx$Date,"%Y"))==year,]
  tmp=tmp[as.numeric(format(tmp$Date,"%m"))==month,]
  write.csv(tmp,tmp.filename,row.names=FALSE)
  if(month<12){
    month=month+1
  }else{
    month=1
    year=year+1
  }
}


####RANDOM FOREST####

#import libraries
library(timeDate)
library(randomForest)

#set options to make sure scientific notation is disabled when writing files
options(scipen=500)

#read in data
dfStore <- read.csv("~/Desktop/walmart/stores.csv")
dfTrain <- read.csv("~/Desktop/walmart/train.csv")
dfTest <- read.csv("~/Desktop/walmart/test.csv")
dfFeatures <- read.csv("~/Desktop/walmart/features.csv")
submission = read.csv("~/Desktop/walmart/sampleSubmission.csv",header=TRUE,as.is=TRUE)


# Merge Type and Size
dfTrainTmp <- merge(x=dfTrain, y=dfStore, all.x=TRUE)
dfTestTmp <- merge(x=dfTest, y=dfStore, all.x=TRUE)


# Merge all the features
train <- merge(x=dfTrainTmp, y=dfFeatures, all.x=TRUE)
test <- merge(x=dfTestTmp, y=dfFeatures, all.x=TRUE)


# Make features for train
train$year = as.numeric(substr(train$Date,1,4))
train$month = as.numeric(substr(train$Date,6,7))
train$day = as.numeric(substr(train$Date,9,10))
train$days = (train$month-1)*30 + train$day
train$Type = as.character(train$Type)
train$Type[train$Type=="A"]=1
train$Type[train$Type=="B"]=2
train$Type[train$Type=="C"]=3
train$IsHoliday[train$IsHoliday=="TRUE"]=1
train$IsHoliday[train$IsHoliday=="FALSE"]=0
train$dayHoliday = train$IsHoliday*train$days
train$logsales = log(4990+train$Weekly_Sales)
#weight certain features more by duplication, not sure if helpful?
train$tDays = 360*(train$year-2010) + (train$month-1)*30 + train$day
train$days30 = (train$month-1)*30 + train$day


#Make features for test
test$year = as.numeric(substr(test$Date,1,4))
test$month = as.numeric(substr(test$Date,6,7))
test$day = as.numeric(substr(test$Date,9,10))
test$days = (test$month-1)*30 + test$day
test$Type = as.character(test$Type)
test$Type[test$Type=="A"]=1
test$Type[test$Type=="B"]=2
test$Type[test$Type=="C"]=3
test$IsHoliday[test$IsHoliday=="TRUE"]=1
test$IsHoliday[test$IsHoliday=="FALSE"]=0
test$dayHoliday = test$IsHoliday*test$days
test$tDays = 360*(test$year-2010) + (test$month-1)*30 + test$day
test$days30 = (test$month-1)*30 + test$day


#Run model
tmpR0 = nrow(submission)
j=1
while (j < tmpR0){
  print(j/tmpR0)#keep track of progress
  #select only relevant data for the store and department tuple
  tmpId = submission$Id[j]
  tmpStr = unlist(strsplit(tmpId,"_"))
  tmpStore = tmpStr[1]
  tmpDept = tmpStr[2]
  dataF1 = train[train$Dept==tmpDept,]
  tmpL = nrow(dataF1[dataF1$Store==tmpStore,])
  #since MAE is weighted, increase weights of holiday data by 5x
  tmpF = dataF1[dataF1$IsHoliday==1,]
  dataF1 = rbind(dataF1,do.call("rbind", replicate(4, tmpF, simplify = FALSE)))
  dataF2 = dataF1[dataF1$Store==tmpStore,]  
  testF1 = test[test$Dept==tmpDept,]
  testF1 = testF1[testF1$Store==tmpStore,]
  testRows = nrow(testF1)
  if (tmpL<10) {#sample size restrictions since rf can fail if there isn't enough data
    #this model uses all dept data (since that store + dept pair does not exist in the training set)
    tmpModel =  randomForest(logsales~Size+Type+ year + month + day + days + dayHoliday + tDays + days30, 
                             ntree=4800, replace=TRUE, mtry=4, data=dataF1)}
  else {
    #this model is trained on store+dept filtered data
    tmpModel =  randomForest(logsales ~ year + month + day + days + dayHoliday + tDays + days30, 
                             ntree=4800, replace=TRUE, mtry=3, data=dataF2)}
  tmpP = exp(predict(tmpModel,testF1))-4990
  k = j + testRows - 1
  submission$Weekly_Sales[j:k] = tmpP
  j = k+1
}


#write the submission to csv for Kaggle submission
write.table(x=submission,
            file='~/Desktop/walmart/randomforest.csv',
            sep=',', row.names=FALSE, quote=FALSE)