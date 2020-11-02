######## R codes for Gait ~ Char + DTI + MRI using LASSO ######


# Import datasets
train_ds = read.csv('Train_set_dti_mri.csv')
test_ds = read.csv('Test_set_dti_mri.csv')

# Extract gait speed data
train_ds_pf = train_ds[, -c(1,2,5,7,8,12)]
test_ds_pf = test_ds[, -c(1,2,5,7,8,12)]

# Import libraries
library(tidyr)
if (!require(caret)) install.packages("caret")
library(caret)
if (!require(glmnet)) install.packages("glmnet")
library(glmnet)
if (!require(gglasso)) install.packages("gglasso")
library(gglasso)
# Drop one NA in the training set
train_ds_pf = train_ds_pf %>% drop_na()
# Drop one NA in the test set
test_ds_pf = test_ds_pf %>% drop_na()


#### Normalising data sets ####
## Specify the last index of numeric var ##
num_n = 266
preProc = preProcess(train_ds_pf[, c(2, 4:num_n)])
train_ds_lasso_std = predict(preProc, train_ds_pf[, c(2, 4:num_n)])
test_ds_lasso_std = predict(preProc, test_ds_pf[, c(2, 4:num_n)])

#### Turn data sets into matrices ####
train_ds_lasso_std = as.matrix(train_ds_lasso_std)
test_ds_lasso_std = as.matrix(test_ds_lasso_std)
train_ds_lasso_y = as.matrix(train_ds_pf[, 3])
test_ds_lasso_y = as.matrix(test_ds_pf[, 3])

## Output mean and std values
preProc_mean = preProc$mean
preProc_std = preProc$std
preProc_cols = colnames(train_ds_pf)[c(2, 4:num_n)]
preProc_summary = data.frame(preProc_cols, preProc_mean, preProc_std, row.names = NULL)
# Output file
# write.csv(preProc_summary, 'gait_lasso_char_dti_mri_summary.csv', row.names = FALSE)




######## Ridge: Char+DTI+MRI (For Adaptive LASSO) ########
#### Ridge model ####
ridge = cv.glmnet(
  train_ds_lasso_std, train_ds_lasso_y, standardize = FALSE,
  family = "gaussian", alpha = 0, intercept = FALSE
)
lambda.ridge = ridge$lambda.min
coef.ridge = matrix(coef(ridge, s = lambda.ridge))




######## LASSO: Char+DTI+MRI ########
# 10 folds (default), no need to standardize (again)
set.seed(21)
lasso = cv.glmnet(
  train_ds_lasso_std, train_ds_lasso_y, standardize = FALSE,
  family = "gaussian", alpha = 1, intercept = FALSE
)
lambda.lasso = lasso$lambda.min

# The first is always intercept regardless "intercept" is specified or not
coef.lasso = matrix(coef(lasso, s = lambda.lasso))[2:(num_n-1)]
# !!FAILED due to no non-zero coefficients




######## Adaptive LASSO: Char+FA+MRI (OLS) ########
gamma_val = 2
coef.ols = solve(t(train_ds_gait_lasso_std) %*% train_ds_gait_lasso_std) %*% t(train_ds_gait_lasso_std) %*% train_ds_gait_lasso_y
# !! FAILED due to small number 2.67895e-20




######## Adaptive LASSO: Char+DTI+MRI (Ridge) ########
gamma_val = 2
wt.ridge = 1 / abs(coef.ridge)**gamma_val

set.seed(21)
ar_lasso = cv.glmnet(
  train_ds_lasso_std, train_ds_lasso_y, standardize = FALSE,
  family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = wt.ridge
)
lambda.ar_lasso = ar_lasso$lambda.min
coef.ar_lasso = matrix(coef(ar_lasso, s = lambda.ar_lasso))[2:(num_n-1)]
# !!FAILED due to no non-zero coefficients




######## Elastic net: Char+DTI+MRI ########
en_cvm = c()
alpha_list = seq(0.01, 0.99, by = 0.01)
for (a in alpha_list){
  set.seed(21)
  en = cv.glmnet(
    train_ds_lasso_std, train_ds_lasso_y, standardize = FALSE,
    family = "gaussian", alpha = a, intercept = FALSE
  )
  # Retrieve the smallest cvm
  en_cvm = c(en_cvm, min(en$en_cvm)) 
}
# Retrieve optimal alpha which gives the smallest cvm
alpha.en = alpha_list[which.min(en_cvm)]
# !! FAILED to find proper tuning alpha 



######## Group Lasso: Char+DTI+MRI #########
dummy_cvt = caret::dummyVars(~ group + gender + vr, data = train_ds_pf)
train_dummy = predict(dummy_cvt, train_ds_pf[,c(1,267,268)])
test_dummy = predict(dummy_cvt, test_ds_pf[,c(1,267,268)])
#### Merge data sets for group lasso ####
train_ds_gglasso = cbind(train_ds_lasso_std, as.matrix(train_dummy))
test_ds_gglasso = cbind(test_ds_lasso_std, as.matrix(test_dummy))
# Create group index
gglasso_groups = c(c(1:264), rep(265, each = 4), rep(266:267, each = 2))

## Group Lasso
# Intercept is set to False for centered data
set.seed(21)
gglasso = cv.gglasso(
  train_ds_gglasso, train_ds_lasso_y, group = gglasso_groups,
  pred.loss = "L1", intercept = FALSE, nfolds = 10, 
)
lambda.gglasso = gglasso$lambda.min # 0.02030409
coef.gglasso = coef(gglasso, s = lambda.gglasso)
coef.gglasso = as.data.frame(coef.gglasso)
coef.gglasso = cbind(Variable = rownames(coef.gglasso), coef.gglasso)
# Output feature selection
# write.csv(coef.gglasso, 'gait_gglasso_woint_char_dti_mri.csv', row.names = FALSE)

gglasso_pred = predict(gglasso, test_ds_gglasso)
caret::RMSE(gglasso_pred, test_ds_lasso_y) # 0.1658985