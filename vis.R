#VISUALISASI
library(tidyverse)
library(DataExplorer)
library(GGally)
library(skimr)
library(ggplot2)

setwd("D:/Postgrad/SEM2/STA582 PMS/PMS Tugas 2")

#plot_intro
#keseluruhan
plot_intro(data = raw, geom_label_args = list(size=2.5))
skim_without_charts(data = raw)
plot_bar(data = raw, by="Re.engagement_result")
barplot(prop.table(table(result))*100, main = "Re-engagement Result", 
        xlab="Persentase (%)", col=c("Steel Blue","Dark Blue"),
        horiz=TRUE)
result = ifelse(raw$Re.engagement_result==1, "Bersedia", "Menolak")

  

#per kelas
yes = raw %>% filter(Re.engagement_result==1)
no = raw %>% filter(Re.engagement_result==0)

plot_bar(data=yes)
plot_bar(data=no)

plot_histogram(data=raw$Age)
hist(raw$Age, xlab="Usia", main="Sebaran Usia Klien ODHA",
     col="Steel Blue")

umur = ifelse(raw$Age<=20, "<20", ifelse(raw$Age>=40, ">40", "20-40"))


hist(no$Age, xlab="Usia", main="Sebaran Usia Klien ODHA yang Menolak Terapi",
     col="coral")
hist(yes$Age, xlab="Usia", main="Sebaran Usia Klien ODHA yang Bersedia Terapi",
     col="pink")

#grafik train b.acc
result0$learner_id = c("Naive Bayes", "Decision Tree", "Random Forest", "kNN",
                       "SVM", "GBM")

result0 %>%
  filter(learner_id %in% c("Naive Bayes", "Decision Tree", "Random Forest", "kNN",
                           "SVM", "GBM")) %>%
  ggplot() +
  geom_boxplot(aes(x=learner_id, y=classif.bacc, colour=learner_id), show.legend=F) +
  ylab("Balanced Accuracy") +
  ggtitle("Train Balanced Accuracy") +
  theme(plot.title = element_text(hjust = 1))+
  theme_bw()

result0 %>%
  filter(learner_id %in% c("Naive Bayes", "Decision Tree", "Random Forest", "kNN",
                           "SVM", "GBM"))%>%
  ggplot() +
  geom_boxplot(aes(x=learner_id, y=classif.sensitivity, fill=learner_id), show.legend=F) +
  ylab("Sensitivity") +
  theme(plot.title = element_text(hjust = 1)) +
  theme_bw()

result0 %>%
  filter(learner_id %in% c("Naive Bayes", "Decision Tree", "Random Forest", "kNN",
                           "SVM", "GBM"))%>%
  ggplot() +
  geom_boxplot(aes(x=learner_id, y=classif.specificity, fill=learner_id), show.legend=F) +
  ylab("Specificity") +
  theme(plot.title = element_text(hjust = 1)) +
  theme_bw()  

#rata2 days from last access orang menolak