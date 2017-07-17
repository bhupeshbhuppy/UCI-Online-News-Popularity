rm(list = ls())
library(reshape2)
library(ggplot2)
library(plotly)
library(ggcorrplot)
library(binr)
library(xgboost)
library(Metrics)
library(MASS)
library(DMwR)

sharing_data<-read.csv("D:/bhupesh/Bootcamp/BootCamp Assignment/OnlineNewsPopularity/OnlineNewsPopularity.csv")
dummy_Features<-c("data_channel_is_lifestyle","data_channel_is_bus",
                  "data_channel_is_socmed","data_channel_is_tech",
                  "data_channel_is_world",
                  "data_channel_is_entertainment","weekday_is_monday",
                  "weekday_is_tuesday","weekday_is_wednesday",
                  "weekday_is_thursday","weekday_is_friday",
                  "weekday_is_saturday","weekday_is_sunday","is_weekend")
conti_features<-colnames(sharing_data)[!(colnames(sharing_data) %in% dummy_Features)]
conti_data<-sharing_data[,conti_features]
#conti_data<-conti_data[,-c(1:2)]
dummy_data<-sharing_data[,dummy_Features]
dummy_data$shares<-sharing_data$shares

###Features Engineering
## removing irrelevent articles

sharing_data<-sharing_data[sharing_data$n_tokens_content !=0,]

## Dividing the features into  continous and categorical
conti_features<-colnames(sharing_data)[!(colnames(sharing_data) %in% dummy_Features)]
conti_data<-sharing_data[,conti_features]
#conti_data<-conti_data[,-c(1:2)]
dummy_data<-sharing_data[,dummy_Features]
dummy_data$shares<-sharing_data$shares

## irrelevent value
conti_data$kw_min_min[conti_data$kw_min_min == -1]<-0

### Binning of data
## n_token_title 13 bins 
additional_features<-c()
cuts<-bins(sharing_data$n_tokens_title,13,minpts =  1, exact.groups = FALSE)
additional_features$n_tokens_title_f<-as.integer(cut(sharing_data$n_tokens_title, bins.getvals(cuts)))
## average_token_length 2 bins
cuts<-bins(sharing_data$average_token_length,2,minpts =  1, exact.groups = FALSE)
additional_features$average_token_length_f<-as.integer(cut(sharing_data$average_token_length, bins.getvals(cuts)))

## num_videos 5 bins
cuts<-bins(sharing_data$num_videos,5,minpts =  1, exact.groups = FALSE)
additional_features$num_videos_f<-as.integer(cut(sharing_data$num_videos, bins.getvals(cuts)))
## num_keywords 8 bin (generated 7)
cuts<-bins(sharing_data$num_keywords,8,minpts =  1, exact.groups = FALSE)
additional_features$num_keywords_f<-as.integer(cut(sharing_data$num_keywords, bins.getvals(cuts)))
## kw_min_min 3 bins (generated 2)
cuts<-bins(sharing_data$kw_min_min,3,minpts =  1, exact.groups = FALSE)
additional_features$kw_min_min_f<-as.integer(cut(sharing_data$kw_min_min, bins.getvals(cuts)))
## kw_max_max 4 bins
cuts<-bins(sharing_data$kw_max_max,4,minpts =  1, exact.groups = FALSE)
additional_features$kw_max_max_f<-as.integer(cut(sharing_data$kw_max_max, bins.getvals(cuts)))
#sharing_data<-cbind(sharing_data,additional_features)
#summary(sharing_data_f1[,c(62:67)])

## Normalization
#normalized_data<-as.data.frame(sapply(conti_data[,-c(1,2)], function(x){LinearScaling(x)})) 

shares<-sharing_data$shares
#newData<-cbind(url = conti_data$url,timedelta= conti_data$timedelta,normalized_data[,-45],dummy_data[-15],additional_features,shares)
newData<-cbind(conti_data[,-47],dummy_data[-15],additional_features,shares)

##subsampling
channel_ent<-newData[newData$data_channel_is_entertainment==1,]
channel_life<-newData[newData$data_channel_is_lifestyle==1,]
channel_bus<-newData[newData$data_channel_is_bus==1,]
channel_tech<-newData[newData$data_channel_is_tech==1,]
channel_world<-newData[newData$data_channel_is_world==1,]
channel_soc<-newData[newData$data_channel_is_socmed==1,]

train<-channel_soc[sample(nrow(channel_soc),as.integer((70*dim(channel_soc)[1])/100),replace=FALSE ),]
test<-channel_soc[!(channel_soc$url) %in% train$url,]

temp_train<-channel_ent[sample(nrow(channel_ent),as.integer((70*dim(channel_ent)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_ent[!(channel_ent$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_world[sample(nrow(channel_world),as.integer((70*dim(channel_world)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_world[!(channel_world$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_tech[sample(nrow(channel_tech),as.integer((70*dim(channel_tech)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_tech[!(channel_tech$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_bus[sample(nrow(channel_bus),as.integer((70*dim(channel_bus)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_bus[!(channel_bus$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_life[sample(nrow(channel_life),as.integer((70*dim(channel_life)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_train<-newData[!(newData$url) %in% train$url & !(newData$url) %in% test$url,]
test<-rbind(test,temp_test)

### Mutual Information using KL
data_kl<-sharing_data
Y_kl <- newData$shares

Y_kl[Y_kl<1400]=-1
Y_kl[Y_kl>=1400]=1

drop<-c('shares','timedelta','url',"n_non_stop_words","title_subjectivity","title_sentiment_polarity")
data_kl <- data_kl[,!names(data_kl) %in% drop ]

data_kl$min_negative_polarity[data_kl$min_negative_polarity<0]=data_kl$min_negative_polarity[data_kl$min_negative_polarity<0]*-1
data_kl$max_negative_polarity[data_kl$max_negative_polarity<0]=data_kl$max_negative_polarity[data_kl$max_negative_polarity<0]*-1
data_kl$avg_negative_polarity[data_kl$avg_negative_polarity<0]=data_kl$avg_negative_polarity[data_kl$avg_negative_polarity<0]*-1


highlow <- function(arg1,med){
  arg1[arg1<=med]=0
  arg1[arg1>med]=1
  
  return(arg1)
}


## using median as cut off
for(i in c("n_tokens_title","n_tokens_content","n_unique_tokens","n_non_stop_unique_tokens","num_hrefs","num_self_hrefs","num_imgs","num_videos","average_token_length","num_keywords","self_reference_min_shares","self_reference_max_shares","self_reference_avg_sharess"))
{
  med <- summary(data_kl[,c(i)])[3]
  data_kl[,c(i)]<-highlow(data_kl[,c(i)],med)
}
## using mean as cut of for the NLP
for(i in c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","global_subjectivity","global_sentiment_polarity","global_rate_positive_words","global_rate_negative_words","rate_positive_words","rate_negative_words","avg_positive_polarity","min_positive_polarity","max_positive_polarity","avg_negative_polarity","min_negative_polarity","max_negative_polarity","abs_title_sentiment_polarity","kw_min_min","kw_min_max","kw_min_avg","kw_max_min","kw_avg_min","kw_avg_avg","kw_avg_max","kw_max_max","kw_max_avg","abs_title_subjectivity"))
{
  med <- summary(data_kl[,c(i)])[4]
  data_kl[,c(i)]<-highlow(data_kl[,c(i)],med)
}




kl<-matrix(nrow = 2, ncol = length(data_kl))

pc0<-length(which(Y_kl==-1))/length(Y_kl)
pc1<-length(which(Y_kl==1))/length(Y_kl)
a<-colSums(data_kl[which(Y_kl==-1),])
b<-colSums(data_kl[which(Y_kl==1),])
c<-colSums(data_kl)/dim(data_kl)[1]
d<-length(which(Y_kl==-1))-colSums(data_kl[which(Y_kl==-1),])
e<-length(which(Y_kl==1))-colSums(data_kl[which(Y_kl==1),])
f<-(dim(data_kl)[1]-colSums(data_kl))/dim(data_kl)[1]
pc01=(pc0*(a/length(which(Y_kl==-1))))/c
pc11=(pc1*(b/length(which(Y_kl==1))))/c
pc00=(pc0*(d/length(which(Y_kl==-1))))/f
pc10=(pc1*(e/length(which(Y_kl==1))))/f
kl[2,]<-pc01*log(pc01/pc0,base = 2)+pc11*log(pc11/pc1,base = 2)
kl[1,]<-pc00*log(pc00/pc0,base = 2)+pc10*log(pc10/pc1,base = 2)

wavg<-((colSums(data_kl)/dim(data_kl)[1])*kl[2,]) +((dim(data_kl)[1]-colSums(data_kl)/dim(data_kl)[1])*kl[1,])


wi1=(c*log(c,base = 2))+(f*log(f,base = 2))


wi=wavg/wi1
z=(1/length(data_kl))*sum(wi)
wifinal=wi/z
k<-as.data.frame(wifinal)


#### Negative Binomial
model_nb<-glm.nb(shares~., train[,-c(1,2)])
test$predition<-predict(model_nb,test[,-c(1,2)])
rmse(test$shares,test$predition)
## 10139.14

## Normalization
normalized_data<-as.data.frame(sapply(conti_data, function(x){LinearScaling(x)})) 
newData_norm<-cbind(url=sharing_data$url, timedelta = sharing_data$timedelta,normalized_data[,-45],dummy_data[-15],additional_features,shares)

##subsampling
channel_ent<-newData_norm[newData_norm$data_channel_is_entertainment==1,]
channel_life<-newData_norm[newData_norm$data_channel_is_lifestyle==1,]
channel_bus<-newData_norm[newData_norm$data_channel_is_bus==1,]
channel_tech<-newData_norm[newData_norm$data_channel_is_tech==1,]
channel_world<-newData_norm[newData_norm$data_channel_is_world==1,]
channel_soc<-newData_norm[newData_norm$data_channel_is_socmed==1,]

train<-channel_soc[sample(nrow(channel_soc),as.integer((70*dim(channel_soc)[1])/100),replace=FALSE ),]
test<-channel_soc[!(channel_soc$url) %in% train$url,]

temp_train<-channel_ent[sample(nrow(channel_ent),as.integer((70*dim(channel_ent)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_ent[!(channel_ent$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_world[sample(nrow(channel_world),as.integer((70*dim(channel_world)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_world[!(channel_world$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_tech[sample(nrow(channel_tech),as.integer((70*dim(channel_tech)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_tech[!(channel_tech$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_bus[sample(nrow(channel_bus),as.integer((70*dim(channel_bus)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_test<-channel_bus[!(channel_bus$url) %in% temp_train$url,]
test<-rbind(test,temp_test)

temp_train<-channel_life[sample(nrow(channel_life),as.integer((70*dim(channel_life)[1])/100),replace=FALSE ),]
train<-rbind(train,temp_train)

temp_train<-newData_norm[!(newData_norm$url) %in% train$url & !(newData_norm$url) %in% test$url,]
test<-rbind(test,temp_test)

#### Negative Binomial Normalized
model_nb<-glm.nb(shares~., train[,-c(1,2)])
test$predition<-predict(model_nb,test[,-c(1,2)])
rmse(test$shares,test$predition)
## 13639.4


## Rpart
rpart_model<-rpart(shares~.,train[,-c(1,2)])
test$Prediction<-as.integer(round(predict(rpart_model, test[,-c(1,2,67,68)])))
rmse(test$shares,test$Prediction)
##13394.41

#### XG bOost
col_name<-colnames(train[,-c(1,2,67)])
model_xg<-xgboost(data        = data.matrix(train[,col_name]),
                  label       = train$shares,
                  nrounds     = 2,
                  objective   = "reg:linear",
                  eval_metric = "rmse")

test$Prediction<-as.integer(round(predict(model_xg, data.matrix(test[,col_name]))))
imp_matrix<-xgb.importance(colnames(train[,col_name]),model_xg)
xgb.plot.importance(imp_matrix)
rmse(test$shares,test$Prediction)
#8630.08

## taking top 10 variables
imp_features<-imp_matrix$Feature[1:10]
train_nb<-train[,imp_features]
train_nb$shares<-train$shares
model_nb<-glm.nb(shares~., train_nb)
## iN SAMPLE Prediction
train$predition<-predict(model_nb,train[,imp_features])
rmse(train$shares,train$predition)
rmse(train$shares,train$predition)/ ( max(train$shares)-min(train$shares) )
## Out of saample Prediction
test$predition<-predict(model_nb,test[,imp_features])
rmse(test$shares,test$predition)
rmse(test$shares,test$predition)/ ( max(test$shares)-min(test$shares) )

#### taking top 5 variables
imp_features<-imp_matrix$Feature[1:4]
train_nb<-train[,imp_features]
train_nb$shares<-train$shares
model_nb<-glm.nb(shares~., train_nb)
summary(model_nb)
test$predition<-predict(model_nb,test[,imp_features])
rmse(test$shares,test$predition)
rmse(test$shares,test$predition)/ ( max(test$shares)-min(test$shares) )
