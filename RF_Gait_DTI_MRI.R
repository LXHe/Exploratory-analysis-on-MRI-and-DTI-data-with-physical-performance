######## R codes for Gait ~ Char + DTI + MRI using Random Forest ######


# Import datasets
train_ds = read.csv('Train_set_dti_mri.csv')
test_ds = read.csv('Test_set_dti_mri.csv')

# Extract gait speed data
train_ds_pf = train_ds[, -c(1,2,5,7,8,12)]
test_ds_pf = test_ds[, -c(1,2,5,7,8,12)]

# Import libraries
library(tidyr)
library(randomForest)
if (!require(caret)) install.packages("caret")
library(caret)
# Drop one NA in the training set
train_ds_pf = train_ds_pf %>% drop_na()
# Drop one NA in the test set
test_ds_pf = test_ds_pf %>% drop_na()




######## With Char+DTI+MRI ########
#### Tuning ####
set.seed(21)
# 10-fold CV
train_control = trainControl(method = "cv", number = 10)
tunegrid = expand.grid(.mtry=c(3:13))
# Specify tuning params
rf_rmse_cv = c()
rf_mtry_cv = c()
for (tree in seq(200, 2000, 200)){
  # Train RF model
  rf_cv = train(
    wksp_1 ~., data = train_ds_pf, method = "rf", tuneGrid = tunegrid, trControl = train_control, ntree = tree
  )
  rf_rmse_cv = c(rf_rmse_cv, min(rf_cv$results$RMSE))
  rf_mtry_cv = c(rf_mtry_cv, rf_cv$bestTune)
}

# Retrieve best parameters
# min(rf_rmse_cv) = 0.2026493
rf_best_idx = which.min(rf_rmse_cv)
# best_idx = 3
rf_best_mtry = c(3:13)[rf_best_idx]
# best_mtry = 5
rf_best_ntree = seq(200, 2000, 200)[rf_best_idx]
# best_ntree = 600

# Plot tree number tuning
plot(
  x = seq(200, 2000, 200), y = rf_rmse_cv, type = "b", xaxt='n',
  xlab = "Number of trees", ylab = "Min CV_RMSE", main = "RF tuning of tree numbers (Char+DTI+MRI selection)"
)
points(x = rf_best_ntree, y = min(rf_rmse_cv), col = 'red', pch = 20)
axis(1,at = seq(200, 2000, 200),labels = seq(200, 2000, 200))


#### Best RF model ####
# RF model with best parameters
rf_opt = randomForest(
  wksp_1 ~., data = train_ds_pf, ntree = rf_best_ntree, 
  mtry = rf_best_mtry, importance = TRUE, type = 'regression'
)

# Predict the test set
rf_pred = predict(rf_opt, test_ds_pf[,-3])
caret::RMSE(rf_pred, test_ds_pf[,3])
# RMSE in the test set: 0.2004289

# Retrieve the importance of each feature
rf_imp = importance(rf_opt)
rf_imp = as.data.frame(rf_imp)
rf_imp = cbind(Variable = rownames(rf_imp), rf_imp)
# Output feature importance
# write.csv(rf_imp, 'gait_rf_imp_char_dti_mri.csv', row.names = FALSE)

# Plot top 10 importance
varImpPlot(
  rf_opt, type = 1, n.var = 10, 
  main = "Top 10 most important variables (Char+DTI+MRI selection)", col = 'blue'
)
abline(v = 3, col = 'red', lty = 2)