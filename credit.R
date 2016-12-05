library(randomForest)
library(caret)
library(e1071)
library(glmnet)
library(ROCR)
data_train = read.csv('train.csv')
data_test = read.csv('test.csv')

#firstly ensure that the data is in good quality
#get idea about the missing value
col_na_train = colSums(is.na(data_train))

col_na_test = colSums(is.na(data_test))
data_train$not.fully.paid=factor(data_train$not.fully.paid, labels=c('no','yes'))

#select important features
payforest = randomForest(not.fully.paid~.,data = data_train,mtry=4,importance = TRUE)
payforest$importance
varImpPlot(payforest)

#from the diagram it seems that credit.policy, int.rate,fico, revl.util,log.annual.inc, revol.bal
#inq.last.6mnths,days.with.cr.line

keep_name = c('not.fully.paid','credit.policy', 'int.rate','fico',  "revol.util" ,'log.annual.inc', "revol.bal","inq.last.6mths","days.with.cr.line")
train_modify = data_train[,names(data_train) %in% keep_name]

modrf<- train(not.fully.paid~.,data = train_modify,method='rf',trControl=trainControl(
                method='cv',classProbs = TRUE))
rf_model = modrf$finalModel
y_hat = predict(rf_model,newdata = data_test[,-14],'prob')

train_final = model.matrix(not.fully.paid~.-1,data=data_train)
grid = 10^seq(10,-2,length = 100)
coef_mat3 = matrix(NA, nrow = 200, ncol =21)
y = data_train$not.fully.paid
for (i in 1:200){
  train = sample(1:nrow(data_train),nrow(data_train)*(9/10)) 
  test = (-train)
  
  lasso.mod3 = glmnet(train_final[train,],y[train],alpha = 1, family = 'binomial',lambda =grid)
  cv.la3 = cv.glmnet(train_final[train,], y[train], alpha = 1, family = 'binomial',lambda = grid)
  best.la3 = cv.la3$lambda.min;
  coef3 = predict(lasso.mod3, s = best.la3, type='coefficients');
  coef_mat3[i,] = as.numeric(abs(coef3)>0.0000001)
}
logic_var = colSums(coef_mat3)>120
selected_name = colnames((train_final[,logic_var[2:length(logic_var)]]))

selected_log_data = data_train[,names(data_train) %in% selected_name]

glm_model = glm(data_train$not.fully.paid~.,family=binomial,data=selected_log_data)


test_log_data = data_test[,colnames(data_test) %in% selected_name]
test_log_data =as.data.frame(test_log_data )

predictTrain_rf = predict(rf_model , newdata =data_test[,-14],'prob')[,2]
ROCRpred_rf = prediction(predictTrain_rf, data_test$not.fully.paid)
ROCRperf_rf = performance(ROCRpred_rf, "tpr", "fpr")


predictTrain = predict(glm_model, newdata =test_log_data, type="response")
ROCRpred = prediction(predictTrain, data_test$not.fully.paid)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf)
plot(ROCRperf_rf,add=TRUE,col = ' red')

