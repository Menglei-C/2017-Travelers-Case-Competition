require(xgboost)
require(methods)
require(data.table)
require(magrittr)
library("ROCR") 
library('caret')
library(clusterSim)
library("Matrix")
library("zipcode")

rm(list=ls())
options(na.action='na.pass')

##############################################################
#clean invalid data
insur=read.csv("Train.csv",header=TRUE,stringsAsFactors=FALSE)
attach(insur)

#age
#unique(ni.age)#NA 216.4974  199.8436  175.5830... because oldest person on record is 122 years old
mark_age=which(ni.age>=122)

#cancel
#unique(cancel)#-1 is incalid input
mark_cancel=which(cancel==-1)

#union all mark_s
markall=NA
markall=union(mark_cancel,mark_age)


#final and cleaned data set
newdata=insur[-markall,]

detach(insur)

##############################################################
#feature engineering

#credit is an ordered categorical variable
newdata$credit=factor(newdata$credit,levels=c("low","medium","high"),ordered = TRUE)
#add new levels to dwelling type since no Landlord in train data
newdata$dwelling.type=factor(newdata$dwelling.type,levels=c("Condo","House","Tenant","Landlord"))

newdata$claim.ind=factor(newdata$claim.ind)
newdata$ni.marital.status=factor(newdata$ni.marital.status)

#add new variables: city and state
data(zipcode)
use_zip=zipcode
newzip=data.frame(zz=as.character(newdata$zip.code))
m=match(newzip$zz,use_zip$zip)
newdata=cbind(newdata,use_zip[m,c(2,3)])
#add all posible city/st as levels
newdata$city=factor(newdata$city,levels=unique(use_zip$city))
newdata$state=factor(newdata$state,levels=unique(use_zip$state))

#drop id, year and zipcode 
newdata=newdata[,-c(1,16,17)]

#split age into groups
newdata[,"age_old_young"]=factor(ifelse(newdata$ni.age<40,"young","old"),ordered=TRUE)
newdata[,"age_round"]=factor(round(newdata$ni.age/10,0),ordered=TRUE)

#family size
newdata$family_zise=newdata$n.adults+newdata$n.children

#ave price=price/#adults
newdata$ave_price=newdata$premium/newdata$n.adults

#age-tenure
newdata$age_tenure=newdata$ni.age-newdata$tenure

###################################################################
#creat training data matrix 
sparse_matrix <- sparse.model.matrix(cancel~.-1, data = newdata)

train=sparse_matrix
train_label=newdata[,"cancel"]
####################################################################
#construct the XGBoost model on training data
bst_mul <- xgboost(data = train, label = train_label, gamma=8,subsample =1,
                   max.depth =15, eta =0.2,min_child_weight =26,
                   nrounds = 30, objective = "binary:logistic",verbose = 0,
                   eval_metric="auc",tree_method="exact")
####################################################################
#plotting feature importance  
importance_matrix <- xgb.importance(dimnames(train)[[2]], model = bst_mul)
xgb.plot.importance(importance_matrix)


####################################################################
#remove useless stored values
rm(insur,newzip,m,mark_age,
   mark_cancel,markall,sparse_matrix,train,train_label,newdata)
###################################################################
##reorganize test data set
options(na.action='na.pass')

test_raw=read.csv("Test.csv",header=TRUE,stringsAsFactors=FALSE)
attach(test_raw)
#age
mark_age=which(ni.age>=122)
#final and cleaned data set
test=test_raw[-mark_age,]

detach(test_raw)

#credit is an ordered categorical variable
test$credit=factor(test$credit,levels=c("low","medium","high"),ordered = TRUE)
#add new levels to dwelling type since no Landlord in train data
test$dwelling.type=factor(test$dwelling.type,levels=c("Condo","House","Tenant","Landlord"))

test$claim.ind=factor(test$claim.ind)
test$ni.marital.status=factor(test$ni.marital.status)


#zip code new levels in test
newzip=data.frame(zz=as.character(test$zip.code))
m=match(newzip$zz,use_zip$zip)
test=cbind(test,use_zip[m,c(2,3)])

#add all posible city/st as levels
test$city=factor(test$city,levels=unique(use_zip$city))
test$state=factor(test$state,levels=unique(use_zip$state))

#drop id, year and zipcode
id=test[,1]
test=test[,-c(1,16,17)]
#newdata$zip.code=factor(newdata$zip.code,levels=use_zip)

#split age into groups
test[,"age_old_young"]=factor(ifelse(test$ni.age<40,"young","old"),ordered=TRUE)
test[,"age_round"]=factor(round(test$ni.age/10,0),ordered=TRUE)

#family size
test$family_zise=test$n.adults+test$n.children

#ave price=price/#adults
test$ave_price=test$premium/test$n.adults

#age-tenure
test$age_tenure=test$ni.age-test$tenure
#######################################################
#creat test data matrix 
sparse_matrix <- sparse.model.matrix(~.-1, data = test)

test_sparse=sparse_matrix
#######################################################
#predict on test data
pred_mul<-predict(bst_mul,test_sparse)
#######################################################
#check the predited result
kk=ifelse(pred_mul>0.5,1,0)
table(kk)
#######################################################
#store the final prediction
pred<-data.frame(id,pred_mul)
names(pred)<-c("id","cancel")
write.csv(pred,"Final.csv")


