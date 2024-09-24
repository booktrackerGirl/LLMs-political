# Library
library(dplyr)
library(stargazer)
library(ltm)
library(patchwork)
library(caret)
devtools::install_github("selva86/InformationValue") 
library(InformationValue)
#install.packages("ISLR")
library(ISLR)
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(ROCR)

# Read the dataset
df = read.csv("./data/entropy_data/hypothesis_data.csv",header=TRUE)
View(df)

################# Point Biserial Correlation ###########################

## Roleplaying
# using '0' as the reference level
biserial.cor(x=df$roleplay_prob, y=df$label_match_roleplay, use = c("all.obs", "complete.obs"), level = 1)
# -0.4708189

## Content
# using '0' as the reference level
biserial.cor(x=df$content_prob, y=df$label_match_content, use = c("all.obs", "complete.obs"), level = 1)
# -0.4133349

## Zero-shot
# using '0' as the reference level
biserial.cor(x=df$zero.shot_prob, y=df$label_match_zero, use = c("all.obs", "complete.obs"), level = 1)
# -0.4491784

## Few-shot
# using '0' as the reference level
biserial.cor(x=df$few.shot_prob, y=df$label_match_few, use = c("all.obs", "complete.obs"), level = 1)
# -0.4340567

## testing through correlation test as well, to get the p-value and significance

df$label_match_roleplay.b <- ifelse(df$label_match_roleplay=="False", 1, 0)
df$label_match_content.b <- ifelse(df$label_match_content=="False", 1, 0)
df$label_match_zero.b <- ifelse(df$label_match_zero=="False", 1, 0)
df$label_match_few.b <- ifelse(df$label_match_few=="False", 1, 0)

cor_1 = cor.test(df$roleplay_prob, df$label_match_roleplay.b, method="pearson") # -0.4708189
cor_1$p.value # 3.670713e-71

cor_2 = cor.test(df$content_prob, df$label_match_content.b, method="pearson") # -0.4133349
cor_2$p.value # 1.16266e-53

cor_3 = cor.test(df$zero.shot_prob, df$label_match_zero.b, method="pearson") # -0.4491784
cor_3$p.value # 3.55257e-64

cor_4 = cor.test(df$few.shot_prob, df$label_match_few.b, method="pearson") # -0.4340567
cor_4$p.value # 1.384703e-59

################# Logistic Regression ###########################

## Roleplaying
# using '0' as the reference level
log_model_roleplay <- glm(label_match_roleplay.b ~ roleplay_prob, data = df, family = "binomial")
summary(log_model_roleplay)
summary(log_model_roleplay)$coefficients
exp(coefficients(log_model_roleplay)[2]) # log-odds

## Content
# using '0' as the reference level
log_model_content <- glm(label_match_content.b ~ content_prob, data = df, family = "binomial")
summary(log_model_content)
summary(log_model_content)$coefficients
exp(coefficients(log_model_content)[2]) 

## Zero-shot
# using '0' as the reference level
log_model_zero <- glm(label_match_zero.b ~ zero.shot_prob, data = df, family = "binomial")
summary(log_model_zero)
summary(log_model_zero)$coefficients
exp(coefficients(log_model_zero)[2]) 

## Few-shot
# using '0' as the reference level
log_model_few <- glm(label_match_few.b ~ few.shot_prob, data = df, family = "binomial")
summary(log_model_few)
summary(log_model_few)$coefficients
exp(coefficients(log_model_few)[2]) 



new_df <- subset(df, select=c(label_match_roleplay.b, roleplay_prob))
colnames(new_df) <- c("Misprediction", "Probability")
new_df$Prompt <- "R"

temp_df <- subset(df, select=c(label_match_content.b, content_prob))
colnames(temp_df) <- c("Misprediction", "Probability")
temp_df$Prompt <- "C"
new_df <- rbind(new_df, temp_df)

temp_df <- subset(df, select=c(label_match_zero.b, zero.shot_prob))
colnames(temp_df) <- c("Misprediction", "Probability")
temp_df$Prompt <- "Z"
new_df <- rbind(new_df, temp_df)

temp_df <- subset(df, select=c(label_match_few.b, few.shot_prob))
colnames(temp_df) <- c("Misprediction", "Probability")
temp_df$Prompt <- "F"
new_df <- rbind(new_df, temp_df)

new_df$Prompt.f <- factor(new_df$Prompt, levels=c("R","C","Z","F"))
#sample_data$x <- relevel(sample_data$x, ref = 4)
new_df <- new_df %>% mutate(id = row_number())

xtabs(~Misprediction + Prompt.f, data = new_df)

## creating the training and test set
set.seed(123)
train <- new_df %>% 
  group_by(Prompt.f) %>%
  dplyr::sample_frac(0.7)

test <- dplyr::anti_join(new_df,
                         train, by = 'id')

#xtabs(~Misprediction + Prompt.f, data = test)

mylogit <- glm(Misprediction ~ Probability + Prompt.f, data = train, family = "binomial")
summary(mylogit)

stargazer(mylogit, align = TRUE, type = "latex")

## CIs using profiled log-likelihood
confint(mylogit)

## odds ratios only
exp(coef(mylogit))

## predictions
predicted <- predict(mylogit, test, type="response")

#find optimal cutoff probability to use to maximize accuracy
optimal <- optimalCutoff(test$Misprediction, predicted)[1]
confusionMatrix(test$Misprediction, predicted)

## AUROC
plotROC(actuals=test$Misprediction, predictedScores=predicted)
sensitivity(actuals=test$Misprediction, predictedScores=predicted)

### K-S plot
pred <- prediction(predicted, test$Misprediction)
perf <- performance(pred,"tpr","fpr")
ks=max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
plot(perf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
print(auc)

ks_plot(actuals=test$Misprediction, predictedScores=predicted) 
        
sensMat <- plotROC(actuals=test$Misprediction, predictedScores=predicted, returnSensitivityMat = TRUE)

# Confidence matrix plot
ctable <- as.table(matrix(c(1253, 184, 38, 53), nrow = 2, byrow = TRUE))
colnames(ctable) <- c('Prediction:\'correct\'', 'Prediction:\'misclassification\'')
rownames(ctable) <- c('Actual:\'correct\'', 'Actual:\'misclassification\'')
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0.95, margin = 1, std = c("margins", "ind.max", "all.max"),
            main = substitute(paste(bold("Confusion Matrix"))),) +
  text(-0.2,0.4, substitute(paste(bold("TN"))), cex=1) + 
  text(0.2, -0.4, substitute(paste(bold("TP"))), cex=1) + 
  text(0.2,0.4, substitute(paste(bold("FN"))), cex=1) + 
  text(-0.2, -0.4, substitute(paste(bold("FP"))), cex=1)

