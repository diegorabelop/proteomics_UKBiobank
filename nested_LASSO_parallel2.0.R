nested_lasso_parallel <- function(data, continuous_vars = NULL, categorical_vars = NULL, outcome_var, 
                                  outer_k = 10, lambda_seq = exp(seq(log(0.001), log(1), length = 10)),
                                  seed = 123, use_class_weights = TRUE) {
  
  # Load necessary libraries
  library(caret)
  library(glmnet)
  library(pROC)
  library(dplyr)
  library(parallel)
  library(doParallel)
  library(ggplot2)
  
  # Detect available cores and create cluster
  num_cores <- detectCores() - 1  # Reserve 1 core for system stability
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  # Define LASSO tuning grid
  lasso_grid <- data.frame(lambda = exp(lambda_seq))
  
  # Create outer folds
  set.seed(seed)
  outer_folds <- createFolds(data[[outcome_var]], k = outer_k, returnTrain = TRUE)
  
  # Initialize results storage
  results <- data.frame(
    Model = character(), Fold = integer(),
    Sensitivity = numeric(), Specificity = numeric(),
    BalancedAccuracy = numeric(), PPV = numeric(), NPV = numeric(),
    AUC = numeric(), AUC_ci_lower = numeric(), AUC_ci_upper = numeric(),
    StartTime = character(), EndTime = character(),
    ElapsedTime = numeric(), Lambda = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Store feature importance across folds
  important_variables_list <- list()
  
  start_time <- Sys.time()
  
  # Loop through outer folds
  for (i in seq_along(outer_folds)) {
    cat("Starting Fold:", i, "\n")
    
    # Split data into training and testing sets
    test_indices <- setdiff(1:nrow(data), outer_folds[[i]])
    train_data <- data[outer_folds[[i]], ]
    test_data <- data[test_indices, ]
    
    # Ensure valid datasets
    if (is.null(train_data) || is.null(test_data) || nrow(train_data) == 0 || nrow(test_data) == 0) {
      stop("Error: One of the training or testing sets is NULL or empty.")
    }
    
    # Handle continuous variables (imputation)
    if (!is.null(continuous_vars) && length(continuous_vars) > 0) {
      preProc_cont <- preProcess(train_data[, continuous_vars, drop = FALSE], method = "knnImpute")
      train_data[, continuous_vars] <- predict(preProc_cont, train_data[, continuous_vars])
      test_data[, continuous_vars] <- predict(preProc_cont, test_data[, continuous_vars])
    }
    
    # Handle categorical variables
    if (!is.null(categorical_vars) && length(categorical_vars) > 0) {
      Mode <- function(x) {
        ux <- unique(x[!is.na(x)])
        if (length(ux) == 0) return(NA)
        ux[which.max(tabulate(match(x, ux)))]
      }
      for (var in categorical_vars) {
        mode_value <- Mode(train_data[[var]])
        train_data[[var]][is.na(train_data[[var]])] <- mode_value
        test_data[[var]][is.na(test_data[[var]])] <- mode_value
        train_data[[var]] <- as.factor(train_data[[var]])
        test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
      }
    }
    
    # Convert target variable to factor
    train_data[[outcome_var]] <- as.factor(train_data[[outcome_var]])
    test_data[[outcome_var]] <- as.factor(test_data[[outcome_var]])
    
    # Compute class weights if enabled
    if (use_class_weights) {
      class_counts <- table(train_data[[outcome_var]])
      class_weights <- 1 / class_counts
      weight_vector <- class_weights[as.character(train_data[[outcome_var]])]
    } else {
      weight_vector <- rep(1, nrow(train_data))
    }
    
    # Hyperparameter tuning for LASSO
    best_model <- NULL
    best_lambda <- NULL
    
    for (params in seq_len(nrow(lasso_grid))) {
      set.seed(seed)
      fit <- cv.glmnet(
        x = as.matrix(model.matrix(as.formula(paste(outcome_var, "~ .")), data = train_data)[, -1]),
        y = as.numeric(train_data[[outcome_var]]),  
        family = "binomial",
        alpha = 1,
        lambda = lasso_grid$lambda,
        weights = weight_vector
      )
      
      best_lambda <- fit$lambda.min
      if (!is.null(fit)) {
        best_model <- fit
      }
    }
    
    # Extract Feature Importance from LASSO
    lasso_coef <- coef(best_model, s = best_lambda)
    lasso_coef_df <- as.data.frame(as.matrix(lasso_coef))
    lasso_coef_df$Feature <- rownames(lasso_coef_df)
    colnames(lasso_coef_df) <- c("Coefficient", "Feature")
    
    # Remove intercept and filter out zero coefficients
    lasso_coef_df <- lasso_coef_df[lasso_coef_df$Coefficient != 0 & lasso_coef_df$Feature != "(Intercept)", ]
    
    # Store feature importance for this fold
    important_variables_list[[i]] <- lasso_coef_df
    
    # Test on Outer Fold
    test_matrix <- as.matrix(model.matrix(as.formula(paste(outcome_var, "~ .")), data = test_data)[, -1])
    if (nrow(test_matrix) == 0) stop("Error: Test matrix is empty.")
    
    final_pred <- as.numeric(predict(best_model, s = best_lambda, newx = test_matrix, type = "response"))
    final_pred_class <- factor(ifelse(final_pred > 0.5, 1, 0), levels = levels(test_data[[outcome_var]]))
    
    # Compute Performance Metrics
    confusion <- confusionMatrix(final_pred_class, test_data[[outcome_var]])
    roc_curve <- roc(test_data[[outcome_var]], final_pred)
    auc_value <- auc(roc_curve)
    ci <- ci.auc(roc_curve)
    
    end_time <- Sys.time()
    elapsed_time <- as.numeric(difftime(end_time, start_time, units = "mins"))
    
    # Store Results
    results <- rbind(results, data.frame(
      Model = "LASSO", Fold = i,
      Sensitivity = as.numeric(confusion$byClass["Sensitivity"]),
      Specificity = as.numeric(confusion$byClass["Specificity"]),
      BalancedAccuracy = as.numeric(confusion$byClass["Balanced Accuracy"]),
      PPV = as.numeric(confusion$byClass["Pos Pred Value"]),
      NPV = as.numeric(confusion$byClass["Neg Pred Value"]),
      AUC = auc_value, AUC_ci_lower = ci[1], AUC_ci_upper = ci[3],
      StartTime = format(start_time), EndTime = format(end_time),
      ElapsedTime = elapsed_time, Lambda = best_lambda
    ))
  }
  
  # Stop parallel cluster
  stopCluster(cl)
  
  # Aggregate feature importance across folds
  all_selected_vars <- do.call(rbind, important_variables_list)
  feature_importance <- aggregate(Coefficient ~ Feature, data = all_selected_vars, FUN = mean)
  feature_importance <- feature_importance[order(abs(feature_importance$Coefficient), decreasing = TRUE), ]
  
  return(list(results = results, feature_importance = feature_importance, predictions = final_pred, true_labels = test_data[[outcome_var]]))
}


final_results_lasso <- nested_lasso_parallel(data, 
                                             continuous_vars = c("Metabolite1", "Metabolite2", "Metabolite3"), 
                                             categorical_vars = c("Category"), 
                                             outcome_var = "Depression")

final_results_lasso
##The most important variables have higher absolute coefficients.
#Zero coefficients mean those features were removed by LASSO as unimportant.
#A positive coefficient increases the probability of the outcome (e.g., Depression = 1).
#A negative coefficient decreases the probability of the outcome.
# Plot the top features
ggplot(final_results_lasso$feature_importance, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Feature Importance in LASSO Model", x = "Features", y = "Coefficient Value") + theme_classic()
# Compute overall ROC curve using all predictions and true labels
rf_roc <- roc(final_results_lasso$true_labels, final_results_lasso$predictions)
print(rf_roc)

# Plot the ROC curve
plot(rf_roc, col = "blue", main = "Random Forest ROC Curve")
