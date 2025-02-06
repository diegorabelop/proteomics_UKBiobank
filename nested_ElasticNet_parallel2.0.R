nested_elastic_net_parallel <- function(data, continuous_vars = NULL, categorical_vars = NULL, outcome_var, 
                                        outer_k = 10, inner_k = 5, 
                                        lambda_seq = exp(seq(log(0.001), log(1), length = 10)),
                                        alpha_seq = seq(0.01, 1, length = 10),  
                                        seed = 123, use_class_weights = TRUE) {
  
  # Load necessary libraries
  library(caret)
  library(glmnet)
  library(pROC)
  library(dplyr)
  library(parallel)
  library(doParallel)
  library(foreach)
  library(ggplot2)
  
  # Detect number of cores and create cluster
  num_cores <- detectCores() - 1  # Reserve 1 core for system stability
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  # Define tuning grid for Elastic Net
  tuning_grid <- expand.grid(lambda = exp(lambda_seq), alpha = alpha_seq)
  
  # Create outer folds for cross-validation
  set.seed(seed)
  outer_folds <- createFolds(data[[outcome_var]], k = outer_k, returnTrain = TRUE)
  
  # Initialize results storage
  results <- data.frame(
    Model = character(), Fold = integer(),
    Sensitivity = numeric(), Specificity = numeric(),
    BalancedAccuracy = numeric(), PPV = numeric(), NPV = numeric(),
    AUC = numeric(), AUC_ci_lower = numeric(), AUC_ci_upper = numeric(),
    StartTime = character(), EndTime = character(),
    ElapsedTime = numeric(), Alpha = numeric(), Lambda = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Initialize a list to store important variables for each fold
  important_variables_list <- list()
  
  start_time <- Sys.time()
  
  # Run outer cross-validation folds in parallel
  for (i in seq_along(outer_folds)) {
    cat("Starting Fold:", i, "\n")
    
    # Split data into training and testing sets
    test_indices <- setdiff(1:nrow(data), outer_folds[[i]])
    train_data <- data[outer_folds[[i]], ]
    test_data <- data[test_indices, ]
    
    # Handle continuous variables (imputation)
    if (!is.null(continuous_vars) && length(continuous_vars) > 0) {
      preProc_cont <- preProcess(train_data[, continuous_vars, drop = FALSE], method = "knnImpute")
      train_data[, continuous_vars] <- predict(preProc_cont, train_data[, continuous_vars])
      test_data[, continuous_vars] <- predict(preProc_cont, test_data[, continuous_vars])
    }
    
    # Handle categorical variables (mode imputation)
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
    
    # Inner loop for hyperparameter tuning
    set.seed(seed)
    inner_folds <- createFolds(train_data[[outcome_var]], k = inner_k, returnTrain = TRUE)
    best_model <- NULL
    best_metric <- -Inf
    best_alpha <- NULL
    best_lambda <- NULL
    
    for (params in seq_len(nrow(tuning_grid))) {
      cv_metrics <- c()
      for (j in seq_along(inner_folds)) {
        inner_train <- train_data[inner_folds[[j]], ]
        inner_test <- train_data[-inner_folds[[j]], ]
        
        set.seed(seed)
        inner_train_matrix <- model.matrix(as.formula(paste(outcome_var, "~ .")), data = inner_train)[, -1]
        inner_test_matrix <- model.matrix(as.formula(paste(outcome_var, "~ .")), data = inner_test)[, -1]
        
        set.seed(seed)
        fit <- cv.glmnet(
          x = as.matrix(inner_train_matrix),
          y = as.numeric(inner_train[[outcome_var]]),
          family = "binomial",
          alpha = tuning_grid$alpha[params],
          lambda = exp(seq(log(0.001), log(1), length = 100)),
          weights = weight_vector[inner_folds[[j]]]
        )
        pred <- as.numeric(predict(fit, as.matrix(inner_test_matrix), type = "response"))
        roc_curve <- roc(inner_test[[outcome_var]], pred)
        cv_metrics <- c(cv_metrics, auc(roc_curve))
      }
      
      if (mean(cv_metrics) > best_metric) {
        best_metric <- mean(cv_metrics)
        best_model <- list(fit = fit, params = tuning_grid[params, ])
        best_alpha <- tuning_grid$alpha[params]
        best_lambda <- tuning_grid$lambda[params]
      }
    }
    
    # Extract non-zero coefficients from the best model
    best_coefs <- coef(best_model$fit, s = best_lambda)
    coef_df <- as.data.frame(as.matrix(best_coefs))
    coef_df$Feature <- rownames(coef_df)
    colnames(coef_df) <- c("Coefficient", "Feature")
    
    # Remove intercept and filter out zero coefficients
    coef_df <- coef_df[coef_df$Coefficient != 0 & coef_df$Feature != "(Intercept)", ]
    
    # Store coefficients for this fold
    important_variables_list[[i]] <- coef_df
    test_matrix <- as.matrix(model.matrix(as.formula(paste(outcome_var, "~ .")), data = test_data)[, -1])
    if (nrow(test_matrix) == 0) stop("Error: Test matrix is empty.")
    
    final_pred <- as.numeric(predict(best_model$fit, s = best_lambda, newx = test_matrix, type = "response"))
    
    final_pred_class <- factor(ifelse(final_pred > 0.5, 1, 0), levels = levels(test_data[[outcome_var]]))
    
    # Compute Performance Metrics
    confusion <- confusionMatrix(final_pred_class, test_data[[outcome_var]])
    roc_curve <- roc(test_data[[outcome_var]], final_pred)
    auc_value <- auc(roc_curve)
    ci <- ci.auc(roc_curve)
    end_time <- Sys.time()
    elapsed_time <- as.numeric(difftime(end_time, start_time, units = "mins"))
    
    # Save results
    results <- rbind(results, data.frame(
      Model = "Elastic Net", Fold = i,
      BalancedAccuracy = as.numeric(confusion$byClass["Balanced Accuracy"]),
      PPV = as.numeric(confusion$byClass["Pos Pred Value"]),
      NPV = as.numeric(confusion$byClass["Neg Pred Value"]),
      AUC = auc_value, AUC_ci_lower = ci[1], AUC_ci_upper = ci[3], Alpha = best_alpha, Lambda = best_lambda,
      StartTime = format(start_time), EndTime = format(end_time),
      ElapsedTime = elapsed_time
    ))
  }
  
  # Stop parallel cluster
  stopCluster(cl)
  
  # Aggregate feature importance across folds
  all_selected_vars <- do.call(rbind, important_variables_list)
  feature_importance <- aggregate(Coefficient ~ Feature, data = all_selected_vars, FUN = mean)
  feature_importance <- feature_importance[order(abs(feature_importance$Coefficient), decreasing = TRUE), ]
  
  return(list(results = results, important_vars = feature_importance, predictions = final_pred, true_labels = test_data[[outcome_var]]))
}



library(pROC)

# Run the function
elastic_results <- nested_elastic_net_parallel(
  data = data,
  continuous_vars = c("Metabolite1", "Metabolite2", "Metabolite3"),
  categorical_vars = "Category",
  outcome_var = "Depression"
)
elastic_results
# Compute ROC for Elastic Net
elastic_roc <- roc(elastic_results$true_labels, elastic_results$predictions)
elastic_roc
# Train another model (e.g., Random Forest)
# Plot ROC curves
plot(elastic_roc, col = "red", main = "ROC Curve Comparison")
# Rename columns for clarity
##The most important variables have higher absolute coefficients.
#Zero coefficients mean those features were removed by LASSO as unimportant.
#A positive coefficient increases the probability of the outcome (e.g., Depression = 1).
#A negative coefficient decreases the probability of the outcome.
# Plot the top features
ggplot(elastic_results$important_vars, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Feature Importance in LASSO Model", x = "Features", y = "Coefficient Value") + theme_classic()

