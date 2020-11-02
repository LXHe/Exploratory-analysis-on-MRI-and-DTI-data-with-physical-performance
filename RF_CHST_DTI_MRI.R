######## R codes for CHST ~ Char + DTI + MRI using Random Forest ######


# Import datasets
train_ds = read.csv('Train_set_dti_mri.csv')
test_ds = read.csv('Test_set_dti_mri.csv')

## Extract SPPB data
train_ds_pf = train_ds[, -c(1,2,5,6,8,12)]
test_ds_pf = test_ds[, -c(1,2,5,6,8,12)]

# Import libraries
library(tidyr)
library(randomForest)
if (!require(caret)) install.packages("caret")
library(caret)
if (!require(ICEbox)) install.packages("ICEbox")
library(ICEbox)
library(dplyr)
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
    chst_1 ~., data = train_ds_pf, method = "rf", tuneGrid = tunegrid, trControl = train_control, ntree = tree
  )
  rf_rmse_cv = c(rf_rmse_cv, min(rf_cv$results$RMSE))
  rf_mtry_cv = c(rf_mtry_cv, rf_cv$bestTune)
}

# Retrieve best parameters
# min(rf_rmse_cv) = 3.12643
rf_best_idx = which.min(rf_rmse_cv)
# rf_best_idx = 2
rf_best_mtry = c(3:13)[rf_best_idx]
# rf_best_mtry = 4
rf_best_ntree = seq(200, 2000, 200)[rf_best_idx]
# rf_best_ntree = 400

# Plot tree number tuning
plot(
  x = seq(200, 2000, 200), y = rf_rmse_cv, type = "b", xaxt='n',
  xlab = "Number of trees", ylab = "Min CV_RMSE", main = "RF tuning of tree numbers (Char+DTI+MRI selection)"
)
points(x = rf_best_ntree, y = min(rf_rmse_cv), col = 'red', pch = 20)
axis(1,at = seq(200, 2000, 200),labels = seq(200, 2000, 200))


#### Best RF model ####
# RF model with best parameters
set.seed(21)
rf_opt = randomForest(
  chst_1 ~., data = train_ds_pf, ntree = rf_best_ntree, 
  mtry = rf_best_mtry, importance = TRUE, type = 'regression'
)

# Predict the test set
rf_pred = predict(rf_opt, test_ds_pf[,-3])
caret::RMSE(rf_pred, test_ds_pf[,3])
# RMSE in the test set: 3.085718

# Retrieve the importance of each feature
rf_imp = importance(rf_opt)
rf_imp = as.data.frame(rf_imp)
rf_imp = cbind(Variable = rownames(rf_imp), rf_imp)
# Output feature importance
# write.csv(rf_imp, 'chst_rf_imp_char_dti_mri.csv', row.names = FALSE)

# Plot top 10 importance
varImpPlot(
  rf_opt, type = 1, n.var = 10, 
  main = "Top 10 most important variables (Char+DTI+MRI selection)", col = 'blue'
)
abline(v = 2.2, col = 'red', lty = 2)


#### MCP_rd_y0 ####
partial1 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'MCP_rd_y0', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary1 = partial1$data
part_mean1 = aggregate(part_summary1$yhat, list(part_summary1$MCP_rd_y0), mean)
colnames(part_mean1) = c('MCP', 'Rel_Change')
# write.csv(part_mean1, 'SPPB_xgb_MCP_cent.csv', row.names = FALSE)


#### ML_R_ad_y0 ####
partial2 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'ML_R_ad_y0', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary2 = partial2$data
part_mean2 = aggregate(part_summary2$yhat, list(part_summary2$ML_R_ad_y0), mean)
colnames(part_mean2) = c('ML', 'Rel_Change')
# write.csv(part_mean2, 'SPPB_xgb_ML_cent.csv', row.names = FALSE)


#### ACR_L_fa_y0 ####
partial3 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'ACR_L_fa_y0', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary3 = partial3$data
part_mean3 = aggregate(part_summary3$yhat, list(part_summary3$ACR_L_fa_y0), mean)
colnames(part_mean3) = c('ACR', 'Rel_Change')
write.csv(part_mean3, 'SPPB_xgb_ACR_cent.csv', row.names = FALSE)


#### Fx/ST_L_rd_y0 ####
partial4 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'Fx.ST_L_rd_y0', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary4 = partial4$data
part_mean4 = aggregate(part_summary4$yhat, list(part_summary4$Fx.ST_L_rd_y0), mean)
colnames(part_mean4) = c('FxST', 'Rel_Change')
# write.csv(part_mean4, 'SPPB_xgb_FxST_cent.csv', row.names = FALSE)


#### left_postcentral ####
partial5 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'left_postcentral', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary5 = partial5$data
part_mean5 = aggregate(part_summary5$yhat, list(part_summary5$left_postcentral), mean)
colnames(part_mean5) = c('LPC', 'Rel_Change')
# write.csv(part_mean5, 'SPPB_xgb_LPC_cent.csv', row.names = FALSE)


#### SLF_R_rd_y0 ####
partial6 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'SLF_R_rd_y0', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary6 = partial6$data
part_mean6 = aggregate(part_summary6$yhat, list(part_summary6$SLF_R_rd_y0), mean)
colnames(part_mean6) = c('SLF', 'Rel_Change')
# write.csv(part_mean6, 'SPPB_xgb_SLF_cent.csv', row.names = FALSE)


#### BMI_1 ####
partial7 = partial(
  rf_opt, plot.engine = 'ggplot2', plot = TRUE, alpha = 0.1,
  pred.var = 'bmi_1', train = train_ds_pf, ice = TRUE, center = TRUE
)
part_summary7 = partial7$data
part_mean7 = aggregate(part_summary7$yhat, list(part_summary7$bmi_1), mean)
colnames(part_mean7) = c('BMI', 'Rel_Change')
# write.csv(part_mean7, 'SPPB_xgb_BMI_cent.csv', row.names = FALSE)