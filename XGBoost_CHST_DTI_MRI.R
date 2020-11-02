######## R codes for SPPB ~ Char + DTI + MRI using XGBoost ######


## Import datasets
train_ds = read.csv('Train_set_dti_mri.csv')
test_ds = read.csv('Test_set_dti_mri.csv')

## Extract SPPB data
train_ds_pf = train_ds[, -c(1,2,5,6,8,12)]
test_ds_pf = test_ds[, -c(1,2,5,6,8,12)]

## Import libraries
library(tidyr)
if (!require(caret)) install.packages("caret")
library(caret)
if (!require(xgboost)) install.packages("xgboost")
library(xgboost)
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)
if (!require(lime)) install.packages("lime")
library(lime)
if (!require(pdp)) install.packages("pdp")
library(pdp)
library(ggplot2)



#### Data preparation ####

# Drop one NA in the training set
train_ds_pf = train_ds_pf %>% drop_na()
# Drop one NA in the test set
test_ds_pf = test_ds_pf %>% drop_na()

# Create dummy variables
dummy_cvt = caret::dummyVars(~ group + gender + vr, data = train_ds_pf)
train_dummy = predict(dummy_cvt, train_ds_pf[,c(1,267,268)])
test_dummy = predict(dummy_cvt, test_ds_pf[,c(1,267,268)])
# Merge data sets for xgb
train_ds_xgb_x= cbind(as.matrix(train_ds_pf[, c(2, 4:266)]), as.matrix(train_dummy))
test_ds_xgb_x = cbind(as.matrix(test_ds_pf[, c(2, 4:266)]), as.matrix(test_dummy))

# Create data set for y
train_ds_xgb_y = as.matrix(train_ds_pf[, 3])
test_ds_xgb_y = as.matrix(test_ds_pf[, 3])


#### Hypertuning ####
## Create hyperparameter grid ##
hyper_grid = expand.grid(
  eta = c(0.01, 0.04, 0.06, 0.1),
  max_depth = c(3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(0.6, 0.8, 1),
  colsample_bytree = c(0.7, 0.8, 0.9, 1),
  optimal_trees = 0,
  min_RMSE = 0
)

for (i in 1:nrow(hyper_grid)){
  # Set hyperparameters
  params_list = list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # Set seed for reproducibility
  set.seed(21)
  
  # Training model: 10 folds CV
  xgb.tuning = xgb.cv(
    params = params_list,
    data = train_ds_xgb_x, 
    label = train_ds_xgb_y, 
    objective = 'reg:squarederror', 
    verbose = 0,
    nfold = 10, 
    early_stopping_rounds = 10, 
    nrounds = 5000
  )
  
  # Retrieve optimal trees and minimal RMSE
  hyper_grid$optimal_trees[i] = which.min(xgb.tuning$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE = min(xgb.tuning$evaluation_log$test_rmse_mean)
}

# Check hypertuning results (top 10)
hyper_grid_sort = hyper_grid %>% dplyr::arrange(min_RMSE)
# 3.198524

# Define the optimal parameter list
# 0.01, 3, 1, 0.6, 0.7, tree(380)
params_opt = list(
  eta = hyper_grid_sort$eta[1],
  max_depth = hyper_grid_sort$max_depth[1],
  min_child_weight = hyper_grid_sort$min_child_weight[1],
  subsample = hyper_grid_sort$subsample[1],
  colsample_bytree = hyper_grid_sort$colsample_bytree[1]
)

set.seed(21)
xgb_final = xgboost(
  params = params_opt,
  data = train_ds_xgb_x, 
  label = train_ds_xgb_y, 
  objective = 'reg:linear', 
  verbose = 0,
  nrounds = 5000
)

xgb_imp = xgb.importance(model = xgb_final)
xgb.plot.importance(
  xgb_imp, top_n = 10, measure = "Gain", 
  xlab = 'Explained variance (Importance)', 
  main = 'Top 10 most important variables (Char+DTI+MRI selection)'
)
# Output feature importance
# write.csv(xgb_imp, 'chst_xgb_imp_char_dti_mri.csv', row.names = FALSE)

# Predicting
xgb_pred = predict(xgb_final, test_ds_xgb_x)
caret::RMSE(xgb_pred, test_ds_xgb_y) # 3.208584


#### Plot important features ####
##  1. Left Fx/ST RD ##
# Non-centered
ice1 = xgb_final %>% 
  partial(
    pred.var = 'Fx.ST_L_rd_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "Chair-stands (s)",
    xlab = "RD of Left Fornix (cres) / Stria terminalis (Fx/ST)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: Chair-stands ~ Fx/ST_L_rd (Non-centered)"
  ) 
plot(ice1)

# Centered
ice2 = xgb_final %>% 
  partial(
    pred.var = 'Fx.ST_L_rd_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of chair-stands (s)",
    xlab = "RD of Left Fornix (cres) / Stria terminalis (Fx/ST)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: Chair-stands ~ Fx/ST_L_rd (Centered)"
  ) 
plot(ice2)
part_summary = ice2$data
part_mean = aggregate(part_summary$yhat, list(part_summary$Fx.ST_L_rd_y0), mean)
colnames(part_mean) = c('Fx.ST', 'Rel_Change')
# write.csv(part_mean, 'CHST_xgb_FXST_cent.csv', row.names = FALSE)


##  2. Baseline BMI ##
# Non-centered
ice3 = xgb_final %>% 
  partial(
    pred.var = 'bmi_1', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "Chair-stands (s)",
    xlab = "BMI"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: Chair-stands ~ BMI (Non-centered)"
  ) 
plot(ice3)


# Centered
ice4 = xgb_final %>% 
  partial(
    pred.var = 'bmi_1', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of Chair-stands (s)",
    xlab = "BMI"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: Chair-stands ~ BMI (Centered)"
  ) 
plot(ice4)
part_summary2 = ice4$data
part_mean2 = aggregate(part_summary2$yhat, list(part_summary2$bmi_1), mean)
colnames(part_mean2) = c('BMI', 'Rel_Change')
# write.csv(part_mean2, 'CHST_xgb_BMI_cent.csv', row.names = FALSE)