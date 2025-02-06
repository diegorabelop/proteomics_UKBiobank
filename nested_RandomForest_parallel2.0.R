# Load required libraries
library(foreach)
library(doParallel)
library(parallel)
library(caret)
library(randomForest)
library(pROC)
library(dplyr)

# Load necessary libraries
library(caret)
library(glmnet)
library(pROC)
library(dplyr)
library(randomForest)
library(parallel)
library(doParallel)

# Load required libraries
library(foreach)
library(doParallel)
library(parallel)
library(caret)
library(randomForest)
library(pROC)
library(dplyr)
install.packages("fastshap")  # SHAP computation for any model
install.packages("shapviz")   # SHAP visualization
library(fastshap)
library(shapviz)


# Load necessary libraries
library(caret)
library(glmnet)
library(pROC)
library(dplyr)
library(randomForest)
library(parallel)
library(doParallel)
library(shapviz)
library(iml)

nested_rf_parallel <- function(data, continuous_vars = NULL, categorical_vars = NULL, outcome_var,
                               outer_k = 10, inner_k = 5, mtry_values = NULL, ntree = 500,
                               seed = 123, use_class_weights = TRUE) {
  
  # If mtry_values is not provided, create a default grid based on the number of predictors
  if (is.null(mtry_values)) {
    predictors <- setdiff(names(data), outcome_var)
    p <- length(predictors)
    default_mtry <- floor(sqrt(p))
    mtry_values <- unique(c(default_mtry, seq(max(1, default_mtry - 2), default_mtry + 2)))
    mtry_values <- mtry_values[mtry_values <= p]
  }
  
  cat("Tuning grid for mtry:", mtry_values, "\n")
  
  # Detect available cores and create a parallel cluster
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  # Set random seed
  set.seed(seed)
  
  # Nested cross-validation setup
  outer_folds <- createFolds(data[[outcome_var]], k = outer_k, returnTrain = TRUE)
  
  # Initialize results data frame
  results <- data.frame(
    Model = character(),
    Fold = integer(),
    Sensitivity = numeric(),
    Specificity = numeric(),
    BalancedAccuracy = numeric(),
    PPV = numeric(),
    NPV = numeric(),
    AUC = numeric(),
    AUC_ci_lower = numeric(),
    AUC_ci_upper = numeric(),
    StartTime = character(),
    EndTime = character(),
    ElapsedTime = numeric(),
    Best_mtry = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Helper function to calculate mode for categorical variables
  Mode <- function(x) {
    ux <- unique(x[!is.na(x)])
    if (length(ux) == 0) return(NA)
    ux[which.max(tabulate(match(x, ux)))]
  }
  #Add a List to Store SHAP Values
  shap_values_list <- list()
  
  # Outer loop for nested cross-validation
  for (i in seq_along(outer_folds)) {
    cat("Starting Fold:", i, "\n")
    start_time <- Sys.time()
    
    test_indices <- setdiff(1:nrow(data), outer_folds[[i]])
    train_data <- data[outer_folds[[i]], ]
    test_data <- data[test_indices, ]
    
    # Handle Continuous Variables
    if (!is.null(continuous_vars) && length(continuous_vars) > 0) {
      set.seed(seed)
      preProc_cont <- preProcess(train_data[, continuous_vars, drop = FALSE], method = "knnImpute")
      train_data[, continuous_vars] <- predict(preProc_cont, train_data[, continuous_vars])
      test_data[, continuous_vars] <- predict(preProc_cont, test_data[, continuous_vars])
    }
    
    # Handle Categorical Variables
    if (!is.null(categorical_vars) && length(categorical_vars) > 0) {
      for (var in categorical_vars) {
        mode_value <- Mode(train_data[[var]])
        train_data[[var]][is.na(train_data[[var]])] <- mode_value
        test_data[[var]][is.na(test_data[[var]])] <- mode_value
      }
      
      for (var in categorical_vars) {
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
    best_model <- list(fit = NULL, model = "Random Forest", params = NA)
    best_metric <- -Inf
    best_mtry <- NA
    
    for (params in seq_along(mtry_values)) {
      cv_metrics <- c()
      
      for (j in seq_along(inner_folds)) {
        inner_train <- train_data[inner_folds[[j]], ]
        inner_test <- train_data[-inner_folds[[j]], ]
        
        set.seed(seed)
        fit <- randomForest(
          x = inner_train[, -which(names(inner_train) == outcome_var)],
          y = inner_train[[outcome_var]],
          mtry = mtry_values[params],
          ntree = ntree
        )
        
        pred <- predict(fit, inner_test, type = "prob")[, 2]
        roc_curve <- roc(inner_test[[outcome_var]], pred)
        cv_metrics <- c(cv_metrics, auc(roc_curve))
      }
      
      if (mean(cv_metrics) > best_metric) {
        best_metric <- mean(cv_metrics)
        best_model <- list(fit = fit, model = "Random Forest", params = mtry_values[params])
        best_mtry <- mtry_values[params]
      }
    }
    
    if (is.null(best_model$fit)) {
      cat("Warning: No valid model found in fold", i, "\n")
      next
    }
    # Extract feature matrix
    X_test <- test_data[, setdiff(names(test_data), outcome_var)]  # Remove target variable
    
    # Create a custom prediction wrapper
    predict_fn <- function(model, newdata) {
      predict(model, newdata, type = "prob")[, 2]  # Probability of class 1
    }
    
    # Compute SHAP values using fastshap
    set.seed(123)
    shap_values <- fastshap::explain(
      best_model$fit,
      X = X_test,
      pred_wrapper = predict_fn,
      nsim = 50  # Number of Monte Carlo simulations
    )
    shap_values_list[[i]] <- shap_values
    
    # Convert to shapviz object for visualization
    sv <- shapviz(shap_values, X = X_test)
    final_pred <- predict(best_model$fit, test_data, type = "prob")[, 2]
    roc_curve <- roc(test_data[[outcome_var]], final_pred)
    auc_value <- auc(roc_curve)
    ci <- ci.auc(roc_curve)
    
    end_time <- Sys.time()
    elapsed_time <- as.numeric(difftime(end_time, start_time, units = "mins"))
    
    # Compute performance metrics
    confusion <- confusionMatrix(as.factor(ifelse(final_pred > 0.5, 1, 0)), test_data[[outcome_var]])
    
    results <- rbind(results, data.frame(
      Model = best_model$model,
      Fold = i,
      Sensitivity = as.numeric(confusion$byClass["Sensitivity"]),
      Specificity = as.numeric(confusion$byClass["Specificity"]),
      BalancedAccuracy = as.numeric(confusion$byClass["Balanced Accuracy"]),
      PPV = as.numeric(confusion$byClass["Pos Pred Value"]),
      NPV = as.numeric(confusion$byClass["Neg Pred Value"]),
      AUC = auc_value,
      AUC_ci_lower = ci[1],
      AUC_ci_upper = ci[3],
      StartTime = format(start_time),
      EndTime = format(end_time),
      ElapsedTime = elapsed_time,
      Best_mtry = best_mtry
    ))
  }
  # Combine SHAP values from all folds
  if (length(shap_values_list) > 0) {
    all_shap_values <- do.call(rbind, shap_values_list)
    
    # Compute mean absolute SHAP value for each feature
    shap_importance <- data.frame(
      Feature = colnames(all_shap_values),
      MeanAbsShap = colMeans(abs(all_shap_values))
    )
    
    # Sort by importance
    shap_importance <- shap_importance[order(shap_importance$MeanAbsShap, decreasing = TRUE), ]
  } else {
    shap_importance <- data.frame(Feature = "No Feature Selected", MeanAbsShap = 0)
  }
  
  # Ensure SHAP values list is not empty
  if (length(shap_values_list) > 0) {
    # Combine SHAP matrices from all folds
    all_shap_values <- do.call(rbind, shap_values_list)
    
    # Ensure SHAP values match feature names
    shap_feature_names <- colnames(all_shap_values)
    
    # Convert into a shapviz object
    sv <- shapviz::shapviz(all_shap_values, X = data[shap_feature_names])
    
  } else {
    sv <- NULL  # No SHAP values found
  }
  
  stopCluster(cl)
  
  
  return(list(results = results, predictions = final_pred,  shap_values = shap_values_list,  # Store SHAP values across folds
              shap_importance = shap_importance,  # Aggregated feature importance 
              shapviz_object = sv,  # SHAP visualization object
              true_labels = test_data[[outcome_var]]))
}

# Example of running the nested random forest function:
# (Replace 'data' with your actual dataset name and specify the appropriate variable names.)
set.seed(123)
data <- data.frame(
  Metabolite1 = c(rnorm(495), rep(NA, 5)),
  Metabolite2 = c(rnorm(490), rep(2, 10)),
  Metabolite3 = c(rnorm(495), rep(10, 5)),
  Category = sample(c("A", "B", "C", NA), 500, replace = TRUE),
  Depression = sample(c(0, 1), 500, replace = TRUE)
)
# Check the first few rows of the updated dataset
head(data)

# View the first few rows of the dataset
head(data)
final_results_rf <- nested_rf_parallel(data,
                                       continuous_vars = c("Metabolite1", "Metabolite2", "Metabolite3"),
                                       categorical_vars = c("Category"),
                                       outcome_var = "Depression",
                                       outer_k = 10, inner_k = 5,
                                       mtry_values = c(1, 2, 3, 4),   # or let the function decide a default grid
                                       ntree = 500,
                                       seed = 123)


# View the fold-by-fold performance results
print(final_results_rf$results)
plot(final_results_rf$shap_values)
sv_importance(final_results_rf$shap_importance, kind = "bee") + theme_classic()
sv_dependence(final_results_rf$shap_importance, v =data)

# Three types of variable importance plots
sv_importance(final_results_rf$shapviz_object)
sv_importance(final_results_rf$shapviz_object, kind = "bar")
sv_importance(final_results_rf$shapviz_object, kind = "both", alpha = 0.2, width = 0.2)
sv_dependence(final_results_rf$shapviz_object, v = "Metabolite1", "Metabolite2")

# Compute overall ROC curve using all predictions and true labels
rf_roc <- roc(final_results_rf$true_labels, final_results_rf$predictions)
print(rf_roc)

# Plot the ROC curve
plot(rf_roc, col = "blue", main = "Random Forest ROC Curve")


# View the fold-by-fold performance results
print(final_results_rf$results)

final_results_rf$predictions

# Compute overall ROC curve using all predictions and true labels
rf_roc <- roc(final_results_rf$true_labels, final_results_rf$predictions)
print(rf_roc)

# Plot the ROC curve
plot(rf_roc, col = "blue", main = "Random Forest ROC Curve")



