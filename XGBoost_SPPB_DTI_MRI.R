######## R codes for SPPB ~ Char + DTI + MRI using XGBoost ######


## Import datasets
train_ds = read.csv('Train_set_dti_mri.csv')
test_ds = read.csv('Test_set_dti_mri.csv')

## Extract SPPB data
train_ds_pf = train_ds[, -c(1,2,5,6,7,12)]
test_ds_pf = test_ds[, -c(1,2,5,6,7,12)]

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
# 1.720585

# Define the optimal parameter list
# 0.01, 3, 1, 0.6, 0.7, tree(621)
# params_opt = list(
#   eta = hyper_grid_sort$eta[1],
#   max_depth = hyper_grid_sort$max_depth[1],
#   min_child_weight = hyper_grid_sort$min_child_weight[1],
#   subsample = hyper_grid_sort$subsample[1],
#   colsample_bytree = hyper_grid_sort$colsample_bytree[1]
# )

params_opt = list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 1,
  subsample = 0.6,
  colsample_bytree = 0.7
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
# write.csv(xgb_imp, 'sppb_xgb_imp_char_dti_mri.csv', row.names = FALSE)

# Predicting
xgb_pred = predict(xgb_final, test_ds_xgb_x)
caret::RMSE(xgb_pred, test_ds_xgb_y) # 1.668515

# Plot out the predicted and actual values
plot(test_ds_xgb_y, round(xgb_pred), type = 'p', col = 'blue', pch = 21, xlab = 'Actual SPPB', ylab = 'Predicted SPPB')
abline(coef = c(0,1))

# Create contingency table
table(test_ds_xgb_y, round(xgb_pred))


#### Plot important features ####
##  1. Right superior cerebellar peduncle AxD ##
# Non-centered
ice1 = xgb_final %>% 
  partial(
    pred.var = 'SCP_R_ad_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "SPPB",
    xlab = "Axial diffusivity (AxD) of Right superior cerebellar peduncle (SCP)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ SCP_R_AxD (Non-centered)"
  ) 
plot(ice1)

# Centered
ice2 = xgb_final %>% 
  partial(
    pred.var = 'SCP_R_ad_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of SPPB",
    xlab = "Axial diffusivity (AxD) of Right superior cerebellar peduncle (SCP)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ SCP_R_AxD (Centered)"
  ) 
plot(ice2)
part_summary = ice2$data
part_mean = aggregate(part_summary$yhat, list(part_summary$SCP_R_ad_y0), mean)
colnames(part_mean) = c('SCP', 'Rel_Change')
# write.csv(part_mean, 'SPPB_xgb_SCP_cent.csv', row.names = FALSE)


##  2. Baseline SCC_ad ##
# Non-centered
ice3 = xgb_final %>% 
  partial(
    pred.var = 'SCC_ad_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "SPPB",
    xlab = "Axial diffusivity (AxD) of Splenium of corpus callosum (SCC)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ SCC_AxD (Non-centered)"
  ) 
plot(ice3)


# Centered
ice4 = xgb_final %>% 
  partial(
    pred.var = 'SCC_ad_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of SPPB",
    xlab = "Axial diffusivity (AxD) of Splenium of corpus callosum (SCC)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ SCC_AxD (Centered)"
  ) 
plot(ice4)
part_summary2 = ice4$data
part_mean2 = aggregate(part_summary2$yhat, list(part_summary2$SCC_ad_y0), mean)
colnames(part_mean2) = c('SCC_ad', 'Rel_Change')
# write.csv(part_mean2, 'SPPB_xgb_SCC_cent.csv', row.names = FALSE)


##  3. Baseline BMI ##
# Non-centered
ice5 = xgb_final %>% 
  partial(
    pred.var = 'bmi_1', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "SPPB",
    xlab = expression(BMI~(kg/m^{2}))
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ BMI (Non-centered)"
  ) 
plot(ice5)

# Centered
ice6 = xgb_final %>% 
  partial(
    pred.var = 'bmi_1', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of SPPB",
    xlab = expression(BMI~(kg/m^{2}))
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ BMI (Centered)"
  ) 
plot(ice6)
part_summary3 = ice6$data
part_mean3 = aggregate(part_summary3$yhat, list(part_summary3$bmi_1), mean)
colnames(part_mean3) = c('BMI', 'Rel_Change')
# write.csv(part_mean3, 'SPPB_xgb_BMI_cent.csv', row.names = FALSE)


##  4. Fx.ST_L_fa_y0 ##
# Non-centered
ice7 = xgb_final %>% 
  partial(
    pred.var = 'Fx.ST_L_fa_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "SPPB",
    xlab = "Fractional anisotropy (FA) of Left Fornix (cres) / Stria terminalis"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ Fx/ST_L_FA (Non-centered)"
  ) 
plot(ice7)

# Centered
ice8 = xgb_final %>% 
  partial(
    pred.var = 'Fx.ST_L_fa_y0', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "Relative change of SPPB",
    xlab = "Fractional anisotropy (FA) of Left Fornix (cres) / Stria terminalis"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ Fx/ST_L_FA (Centered)"
  ) 
plot(ice8)
part_summary4 = ice8$data
part_mean4 = aggregate(part_summary4$yhat, list(part_summary4$Fx.ST_L_fa_y0), mean)
colnames(part_mean4) = c('RFP', 'Rel_Change')
# write.csv(part_mean4, 'SPPB_xgb_FxST_cent.csv', row.names = FALSE)


##  5. Left rostral anterior cingulate ##
# Non-centered
ice9 = xgb_final %>% 
  partial(
    pred.var = 'left_rostralanteriorcingulate', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = FALSE, 
    ylab = "SPPB",
    xlab = "Left rostral anterior cingulate thickness (mm)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ Left rostral anterior cingulate (Non-centered)"
  ) 
plot(ice9)

# Centered
ice10 = xgb_final %>% 
  partial(
    pred.var = 'left_rostralanteriorcingulate', train = train_ds_xgb_x, ice = TRUE
  ) %>%
  autoplot(
    rug = TRUE, train = train_ds_xgb_x, alpha = 0.1, center = TRUE, 
    ylab = "SPPB",
    xlab = "Left rostral anterior cingulate thickness (mm)"
  ) + 
  ggtitle(
    "Individual Conditional Expectation: SPPB ~ Left rostral anterior cingulate (Centered)"
  ) 
plot(ice10)
part_summary5 = ice10$data
part_mean5 = aggregate(part_summary5$yhat, list(part_summary5$left_rostralanteriorcingulate), mean)
colnames(part_mean5) = c('LRAC', 'Rel_Change')
# write.csv(part_mean5, 'SPPB_xgb_LRAC_cent.csv', row.names = FALSE)