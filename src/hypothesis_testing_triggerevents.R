# Library
library(dplyr)
library(stargazer)
library(rstatix)
library(tidyverse)
library(ggplot2)

df = read.csv("./data/sexist_political_metadata.csv",header=TRUE)
#View(df)

#Check if the variables are factors and ordinal or not
is.factor(df$label) #FALSE
# But we do not need to factorize it further as already binary.

is.factor(df$trigger_event) #FALSE
df$trigger_event.f <- ordered(df$trigger_event, levels = c("political", "sexist"))
# Levels: normal < political < sexist (factorized in ordered form)
is.factor(df$trigger_event.f) #TRUE

# Now we can use it in our model
model = wilcox_test(label~trigger_event.f, data= df, 
                     exact = TRUE, alternative = "two.sided")%>%
  adjust_pvalue(method = "BH") %>%
  add_significance("p.adj")
# A tibble: 1 × 9
# .y.   group1    group2    n1    n2 statistic        p    p.adj p.adj.signif
# <chr> <chr>     <chr>  <int> <int>     <dbl>    <dbl>    <dbl> <chr>       
#   1 label political sexist   566   707   186136. 0.000011 0.000011 ****  


df %>% wilcox_effsize(label~trigger_event.f)
# A tibble: 1 × 7
# .y.   group1    group2 effsize    n1    n2 magnitude
# * <chr> <chr>     <chr>    <dbl> <int> <int> <ord>    
#   1 label political sexist   0.123   566   707 small  
# 2 label normal    sexist    0.107     292   707 small    
# 3 label political sexist    0.106     274   707 small 


