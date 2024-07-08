# Library
library(dplyr)
library(stargazer)
library(ltm)

# Read the dataset
df = read.csv("./data/entropy_data/hypothesis_data.csv",header=TRUE)
View(df)

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


