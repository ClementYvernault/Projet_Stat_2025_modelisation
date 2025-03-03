##### Packages et fonctions requis #####

#### Packages ####

# Base et dataviz
library(knitr)
library(kableExtra)
library(DT)
library(skimr)
library(OneR)
library(table1)
library(colorspace)
library(ggplot2)
library(gridExtra)
library(GGally)
library(ggforce)

# Meta-packages
library(tidymodels)
library(tidyverse)
library(learntidymodels)
library(recipes)
library(parsnip)
library(workflows)
library(vip)

# Modèles
library(glmnet)
library(rules)
library(kknn)
library(ranger)
library(xgboost)
library(rpart)
library(nnet)
library(NeuralNetTools)
library(kernlab)
library(sda)

# Facilitateurs
library(tune)
library(finetune)
library(corrr)
library(baguette)
library(DALEXtra)
library(pdp)

# Autres
install.packages("mixOmics")
library(mixOmics)

# Pour éviter les conflits entre packages (c'est tidymodels qui gagne !)
tidymodels_prefer()

#### Fonctions ####
rm(list=ls())
pvalue <- function(x, ...) {
  x <- x[-length(x)]  # Remove "overall" group
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  if (is.numeric(y)) { # For numeric variables, perform an ANOVA
    p <- summary(aov(y ~ g))[[1]][["Pr(>F)"]][1]
  } else { # For categorical variables, perform a chi-squared test
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-value, using an HTML entity for the less-than sign.
  c("", sub("<", "&lt;", format.pval(p)))
}



plot_validation_results <- function(recipe, dat = data_test) {
  recipe %>%
    recipes::prep() %>%
    bake(new_data = dat) %>%
    ggplot(aes(x = .panel_x, y = .panel_y, color = y, fill = y)) +
    geom_point(alpha = 0.4, size = 0.5) +
    geom_autodensity(alpha = .3) +
    facet_matrix(vars(- y), layer.diag = 2) + 
    scale_color_brewer(palette = "Dark2") + 
    scale_fill_brewer(palette = "Dark2")
}

transformer_en_dummy <- function(data) {
  # Parcourir chaque colonne du data frame
  for (col in names(data)) {
    if (is.factor(data[[col]]) && col != "y") {
      # Calculer la modalité la plus fréquente
      freq <- table(data[[col]], useNA = "no")
      mode <- names(freq)[which.max(freq)]  # Modalité la plus présente
      
      # Créer les dummy variables pour cette colonne
      dummies <- model.matrix(~ data[[col]] - 1)  # -1 pour exclure l'intercept
      
      # Supprimer la colonne correspondant à la modalité la plus fréquente
      dummies <- dummies[, !colnames(dummies) %in% paste0("data[[col]]", mode), drop = FALSE]
      
      # Ajouter les nouvelles colonnes au data frame
      colnames_dummies <- gsub("data\\[\\[col\\]\\]", paste0(col, "_"), colnames(dummies))
      colnames(dummies) <- colnames_dummies
      data <- cbind(data, dummies)
      
      # Supprimer la colonne factor originale
      data[[col]] <- NULL
    }
  }
  return(data)
}

##### Exploration des données #####

#### Données et objectifs ####

data <- readRDS("data_post_etape_4_Mean.Rdata")
data <- data[,-1]
data <- transformer_en_dummy(data)
DT:::datatable(head(data), caption = "Table 1: Données")

#### Résumé des variables ####

skim(data)

#### Analyse bivariée ####

GGally:::ggpairs(data, aes(color = y))



table1( ~ . | y, data = data, extra.col = list(`p-value`= pvalue), extra.col.pos = 1)

#### Analyse multivariée ####

data_split <- rsample:::initial_split(data, strata = y, prop = 0.75)
data_train <- training(data_split)  # 75% des observations
data_test  <- testing(data_split)   # 25% des observations

normalized_rec <- 
  recipes::recipe(y ~ ., data = data_train) %>% 
  step_normalize(- y)

normalized_rec %>%
  step_pls(all_numeric_predictors(), outcome = "y", num_comp = 3) %>%
  plot_validation_results() + 
  ggtitle("Discriminant Partial Least Squares")



normalized_rec %>%
  step_pls(all_numeric_predictors(), outcome = "y", num_comp = 3) %>%
  recipes::prep() %>% 
  plot_top_loadings(component_number <= 3, n = 3, type = "pls") + 
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Discriminant Partial Least Squares")

##### Validation croisée #####

#### Séparation des observations ; package {rsample} ####

set.seed(1042)
data_split <- rsample:::initial_split(data, strata = y, prop = 0.75)
data_train <- training(data_split)  # 75% des observations
data_test  <- testing(data_split)   # 25% des observations
data_folds <- rsample:::vfold_cv(data_train, strata = y, v = 3, repeats = 3)

#### Pré-traitement des variables ; package {recipes} ####

normalized_rec <- 
  recipe(y ~ ., data = data_train) %>% 
  step_normalize(- y)

##### Modèles évalués #####

#### Spécifications ; package {parsnip} ####

multinom_reg_spec <- multinom_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

nearest_neighbor_spec <- 
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

rand_forest_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

boost_tree_spec <- boost_tree(tree_depth = tune(), trees = 1000, learn_rate = tune(), mtry = tune(), min_n = tune(), loss_reduction = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

decision_tree_spec <- decision_tree(cost_complexity = tune(), tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification") 

bag_tree_spec <- bag_tree() %>% 
  set_engine("rpart", times = 50L) %>% 
  set_mode("classification")

C5_rules_spec <- C5_rules() %>%
  set_engine("C5.0") %>%
  set_mode("classification")

mlp_spec <- mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
  set_engine("nnet", MaxNWts = 2600) %>% 
  set_mode("classification")

svm_rbf_spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

svm_poly_spec <- svm_poly(cost = tune(), degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

#### Application ; package {workflows} ####

# Application des modèles (basés sur des distances) sur les variables explicatives normalisées
normalized <- workflow_set(
  preproc = list(normalized = normalized_rec), 
  models  = list(multinom_reg     = multinom_reg_spec, 
                 nearest_neighbor  = nearest_neighbor_spec, 
                 mlp               = mlp_spec, 
                 svm_rbf           = svm_rbf_spec, 
                 svm_poly          = svm_poly_spec))

# Application des modèles (non basés sur des distances) sur les variables explicatives non-normalisées
no_pre_proc <- workflow_set(
  preproc = list(simple = recipe(y ~ ., data = data_train)),
  models  = list(bag_tree     = bag_tree_spec, 
                 C5_rules      = C5_rules_spec,
                 decision_tree = decision_tree_spec, 
                 rand_forest   = rand_forest_spec, 
                 boost_tree    = boost_tree_spec))

# Assemblage des deux workflows
all_workflows <- bind_rows(no_pre_proc, normalized) %>% 
  mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))

#### Paramètrage ; package {tune} ####

grid_ctrl <- control_grid(
  save_pred     = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE)

workflows_res <- all_workflows %>%
  workflowsets:::workflow_map(
    seed      = 42,
    resamples = data_folds,
    grid      = 25,
    metrics   = metric_set(accuracy, average_precision, kap),
    control   = grid_ctrl)

#### Performance ; package {yardstick} ####

### Qualité de modélisation ###

workflows_res %>% 
  rank_results() %>% 
  select(model, .config, kap = mean, rank)


chisq.test(data$y, data$A04_My10Mfloc_modéré)
autoplot(workflows_res,
         rank_metric = "kap",    # how to order models
         select_best = TRUE)  +  # one point per workflow
  geom_text(aes(y = mean - 0.02, label = wflow_id), angle = 90, hjust = 1) + theme(legend.position = "none") + ggtitle("Qualité de modèlisation")



Perf_Model.all <-
  workflows_res %>% 
  rank_results(rank_metric = "kap", select_best = TRUE) %>% 
  select(rank, model, .config, .metric, n, mean, std_err)

Model.rank <- unique(Perf_Model.all$model)
Nb.Model   <- length(Model.rank)

### Qualité de prédiction ###

bestmodel.tuned        <- list()
fitted.bestmodel.tuned <- list()

for (k in 1 : Nb.Model){
  
  # Sélection des meilleurs modèles (meilleures valeurs des hyper-paramètres)
  bestmodel.tuned[[k]] <- 
    workflows_res %>% 
    extract_workflow_set_result(Model.rank[k]) %>% 
    select_best(metric = "kap")
  
  # Application de ces modèles aux données tests
  fitted.bestmodel.tuned[[k]] <- 
    workflows_res %>%
    extract_workflow(Model.rank[k]) %>%
    finalize_workflow(bestmodel.tuned[[k]]) %>% 
    last_fit(data_split)
}
names(bestmodel.tuned) <- names(fitted.bestmodel.tuned) <- Model.rank

Perf_Pred.all <- list()
for (k in 1 : Nb.Model){
  Perf_Pred.all[[k]] <- collect_metrics(fitted.bestmodel.tuned[[k]]) %>% 
    add_column(model = Model.rank[k], .before = T) %>%
    select(model, .metric, .estimate) %>%
    spread(key = .metric, value = .estimate) %>%
    rename(Accuracy.p = accuracy)  %>%
    rename(roc_auc = roc_auc)%>%
    mutate(across(where(is.numeric), round, 3))
}
Perf_Pred <- bind_rows(Perf_Pred.all)

### Valeurs des hyper-paramètres optimisés ###

bestmodel.tuned

### Par modalité de la variable à expliquer ###

Perf_Pred_class.list <- list()
for (k in 1 : Nb.Model){
  pred.table <- fitted.bestmodel.tuned[[k]] %>% 
    collect_predictions() %>%
    conf_mat(truth = y, estimate = .pred_class)
  Perf_Pred_class.list[[k]] <- diag(pred.table$table)/colSums(pred.table$table)*100
}

Perf_Pred_class <- bind_rows(Perf_Pred_class.list)  %>% 
  add_column(model = Model.rank, .before = T) %>%
  mutate(across(where(is.numeric), round, 2))
Perf_Pred_class

### Synthèse ###

Perf_Model <- Perf_Model.all %>%
  select(model, rank, .metric, mean) %>%
  spread(key = .metric, value = mean) %>%
  rename(Accuracy.m = accuracy)  %>%
  rename(Precision = average_precision) %>%
  rename(Kappa = kap) %>%
  mutate(across(where(is.numeric), round, 3))

Perf <- left_join(Perf_Model, Perf_Pred, by = 'model') %>%
  left_join(., Perf_Pred_class, by = 'model')  %>%
  arrange(rank)

kbl(Perf) %>%
  kable_classic(full_width = F, position = "left") %>%
  add_header_above(c(" " = 1, "Qualité de modélisation" = 4, "Qualité de prédiction" = 6))

##### Interprétation des modèles ; packages {DALEXtra} et {pdp} #####

mod               <- list()
explainer         <- list()
vip.tab           <- list()
vip.plot          <- list()
profil            <- list()
profil.plot       <- list()

for (k in 1 : Nb.Model){
  mod[[k]] <- workflows_res %>%
    extract_workflow(Model.rank[k]) %>%
    finalize_workflow(bestmodel.tuned[[k]]) %>% 
    last_fit(data_split)
  
  explainer[[k]] <- explain_tidymodels(extract_fit_parsnip(mod[[k]]), data = recipe(y ~ ., data = data_train) %>%
                                         recipes::prep() %>% 
                                         bake(new_data = NULL, all_predictors()),
                                       y = data_train$y, verbose = F)
  vip.tab[[k]]     <- DALEX:::model_parts(explainer[[k]], type = "variable_importance")
  vip.plot[[k]]    <- plot(DALEX:::model_parts(explainer[[k]], type = "variable_importance"), max_vars = 20, title = paste(paste(Model.rank[k], '/ Acc.=', sep = ' ', Perf$Accuracy.m[k])), subtitle = '')
  profil[[k]]       <- model_profile(explainer[[k]], type = "partial", variables = colnames(data)[-5])
  profil.plot[[k]] <- plot(profil[[k]], variables = colnames(data)[-5]) + ggtitle(paste(paste(Model.rank[k], '/ Acc.=', sep = ' ', Perf$Accuracy.m[k])), "")
}

#### Importance de chaque variable ####

lapply(1 : Nb.Model, function(k) vip.plot[[k]])

#### Influence de chaque variable ####

lapply(1 : Nb.Model, function(k) profil.plot[[k]])

##### Utilisation des modèles (prédiction) #####

lapply(1 : Nb.Model, function(k) explainer[[k]] %>% DALEX:::predict_parts(new_observation = data_test[1, ]) %>% plot()  + ggtitle(paste(paste(Model.rank[k], '/ Acc.=', sep = ' ', Perf$Accuracy.m[k])), ""))