library(tidyverse)
library(mlr3verse)
library(mlr3tuning)
library(mlr3extralearners)
library(mlr3)
library(mlr3learners)
library(caret)
library(e1071) 
library(ranger) 
library(kknn) 
library(kernlab) 
library(gbm) 

# Import data
raw = read.csv("dataset.csv")
raw[,4] = as.Date(raw[,4], "%m/%d/%y")
raw[,5] = as.Date(raw[,5], "%m/%d/%y")
raw[,24] = as.Date(raw[,24], "%m/%d/%y")

u = c(2,7:22,25)
for (i in u) {
  raw[,i] = as.factor(raw[,i])
}

# Data Splitting
set.seed(1000)
idx = sample.int(nrow(raw), size=0.7*nrow(raw))
train = raw[idx, c(2,3,6:23,25)]
test = raw[-idx, c(2,3,6:23,25)]

# Data training using mlr3 ecosystem
task_class = TaskClassif$new(id="class", backend = train, 
                             target = "Re.engagement_result",
                             positive = "1")

model_nb = lrn("classif.naive_bayes")
model_dt = lrn("classif.rpart")
model_rf = lrn("classif.ranger")
model_knn = lrn("classif.kknn", k=10, kernel="rectangular")
model_svm = lrn("classif.ksvm")
model_gbm = lrn("classif.gbm")
as.data.table(model_svm$param_set) #hyperparam checking

# Define Hyperparameter
param_bound_nb = ParamSet$new(params = list(ParamDbl$new("laplace", 
                                                         lower = 0,
                                                         upper = 1)))
param_bound_dt = ParamSet$new(params = list(ParamDbl$new("cp", 
                                                         lower = 0,
                                                         upper = 1),
                                            ParamInt$new("minsplit",
                                                         lower = 1,
                                                         upper = 5)))
param_bound_rf = ParamSet$new(params = list(ParamInt$new("max.depth", 
                                                         lower = 1,
                                                         upper = 5)))
param_bound_knn = ParamSet$new(params = list(ParamInt$new("k",
                                                          lower = 1,
                                                          upper = 10)))
param_bound_svm = ParamSet$new(params = list(ParamInt$new("degree",
                                                          lower = 1,
                                                          upper = 4),
                                             ParamFct$new("kernel",
                                                          levels="polydot")))
param_bound_gbm = ParamSet$new(params = list(ParamInt$new("n.minobsinnode",
                                                          lower = 1,
                                                          upper = 5)))

# CV inner validation
terminate = trm("evals", n_evals = 10) 
tuner = tnr("random_search") 
resample_inner = rsmp("holdout")

# Autotuner for hyperparameter optimization
model_nb_tune = AutoTuner$new(learner = model_nb,
                              measure = msr("classif.bacc"),
                              terminator = terminate,
                              resampling = resample_inner,
                              search_space = param_bound_nb,
                              tuner = tuner,
                              store_models = TRUE)

model_dt_tune = AutoTuner$new(learner = model_dt,
                              measure = msr("classif.bacc"),
                              terminator = terminate,
                              resampling = resample_inner,
                              search_space = param_bound_dt,
                              tuner = tuner,
                              store_models = TRUE)

model_rf_tune = AutoTuner$new(learner = model_rf,
                              measure = msr("classif.bacc"),
                              terminator = terminate,
                              resampling = resample_inner,
                              search_space = param_bound_rf,
                              tuner = tuner,
                              store_models = TRUE)

model_knn_tune = AutoTuner$new(learner = model_knn,
                               measure = msr("classif.bacc"),
                               terminator = terminate,
                               resampling = resample_inner,
                               search_space = param_bound_knn,
                               tuner = tuner,
                               store_models = TRUE)

model_svm_tune = AutoTuner$new(learner = model_svm,
                               measure = msr("classif.bacc"),
                               terminator = terminate,
                               resampling = resample_inner,
                               search_space = param_bound_svm,
                               tuner = tuner,
                               store_models = TRUE)

model_gbm_tune = AutoTuner$new(learner = model_gbm,
                               measure = msr("classif.bacc"),
                               terminator = terminate,
                               resampling = resample_inner,
                               search_space = param_bound_gbm,
                               tuner = tuner,
                               store_models = TRUE)

# CV outer validation
resample_outer = rsmp("cv", folds=5)
set.seed(1)
resample_outer$instantiate(task = task_class)

model_class = list(model_nb_tune,
                   model_dt_tune,
                   model_rf_tune,
                   model_knn_tune,
                   model_svm_tune,
                   model_gbm_tune)

design = benchmark_grid(tasks = task_class,
                        learners = model_class,
                        resamplings = resample_outer)

lgr::get_logger("bbotk")$set_threshold("warn")
bmr = benchmark(design,store_models = TRUE)

result = bmr$aggregate(list(msr("classif.acc"), 
                            msr("classif.specificity"),
                            msr("classif.sensitivity"),
                            msr("classif.bacc")))
result
result_fin = as.data.frame(result[1:6, c(4,7:10)])

#Showing best hyperparameter on each model
get_param_res = function(i){
  as.data.table(bmr)$learner[[i]]$tuning_result
}

# Optimized hyperparameter of Naive Bayes
best_nb_param =map_dfr(1:5,get_param_res)
best_nb_param

best_nb_param %>% slice_max(classif.bacc) 

best_nb_param_value <-  c(best_nb_param %>%
                            slice_max(classif.bacc) %>%
                            pull(laplace))

#Optimized hyperparameter of Decision Tree
best_dt_param =map_dfr(6:10,get_param_res)
best_dt_param

best_dt_param %>% slice_max(classif.bacc)

best_dt_param_value <-  c(best_dt_param %>%
                            slice_max(classif.bacc) %>%
                            pull(cp),
                          best_dt_param %>%
                            slice_max(classif.bacc) %>%
                            pull(minsplit))

#Optimized hyperparameter of Random Forest
best_rf_param =map_dfr(11:15,get_param_res)
best_rf_param

best_rf_param %>% slice_max(classif.bacc)

best_rf_param_value <-  c(best_rf_param %>%
                            slice_max(classif.bacc) %>%
                            pull(max.depth))

#Optimized hyperparameter of K-Nearest Neighbor
best_knn_param =map_dfr(16:20,get_param_res)
best_knn_param

best_knn_param %>% slice_max(classif.bacc)

best_knn_param_value <-  c(best_knn_param %>%
                            slice_max(classif.bacc) %>%
                            pull(k))

#Optimized hyperparameter of Support Vector Machine
best_svm_param =map_dfr(21:25,get_param_res)
best_svm_param

best_svm_param %>% slice_max(classif.bacc)

best_svm_param_value <-  c(best_svm_param %>%
                             slice_max(classif.bacc) %>%
                             pull(degree))

#Optimized hyperparameter of Gradient Boosting 
best_gbm_param =map_dfr(26:30,get_param_res)
best_gbm_param

best_gbm_param %>% slice_max(classif.bacc)

best_gbm_param_value <-  c(best_gbm_param %>%
                             slice_max(classif.bacc) %>%
                             pull(n.minobsinnode))

# Data Testing

# Naive Bayes
model_nb_best = lrn("classif.naive_bayes", laplace=best_nb_param_value)
model_nb_best$train(task = task_class)

prediction_nb <- model_nb_best$predict_newdata(newdata = test)
as.data.table(prediction_nb)
nb = model_nb_best$predict_newdata(newdata = test)$confusion
confusionMatrix(nb, positive="1")

# Decision Tree
model_dt_best = lrn("classif.rpart", cp=best_dt_param_value[1],
                    minsplit=best_dt_param_value[2])
model_dt_best$train(task = task_class)

prediction_dt <- model_dt_best$predict_newdata(newdata = test)
as.data.table(prediction_dt)
dt = model_dt_best$predict_newdata(newdata = test)$confusion
confusionMatrix(dt, positive="1")

# Random Forest
model_rf_best = lrn("classif.ranger", max.depth=best_rf_param_value, 
                    importance="impurity")
model_rf_best$train(task = task_class)

prediction_rf <- model_rf_best$predict_newdata(newdata = test)
as.data.table(prediction_rf)
rf = model_rf_best$predict_newdata(newdata = test)$confusion
confusionMatrix(rf, positive="1")

# KNN
model_knn_best = lrn("classif.kknn", k=best_knn_param_value,
                     kernel="rectangular")
model_knn_best$train(task = task_class)

prediction_knn <- model_knn_best$predict_newdata(newdata = test)
as.data.table(prediction_knn)
knn = model_knn_best$predict_newdata(newdata = test)$confusion
confusionMatrix(knn, positive="1")

# SVM
model_svm_best = lrn("classif.ksvm", degree=best_svm_param_value, 
                     kernel="polydot")
model_svm_best$train(task = task_class)

prediction_svm <- model_svm_best$predict_newdata(newdata = test)
as.data.table(prediction_svm)
svm = model_svm_best$predict_newdata(newdata = test)$confusion
confusionMatrix(svm, positive="1")

# GBM
model_gbm_best = lrn("classif.gbm", n.minobsinnode=best_gbm_param_value)
model_gbm_best$train(task = task_class)
model_gbm_best$importance()

prediksi_gbm <- model_gbm_best$predict_newdata(newdata = test)
as.data.table(prediksi_gbm)
gbm = model_gbm_best$predict_newdata(newdata = test)$confusion
confusionMatrix(gbm, positive="1")

#Best result is GBM
