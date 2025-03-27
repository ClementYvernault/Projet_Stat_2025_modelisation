
rm(list=ls())



stepwise_multinomial_vglm <- function(data, response_var, alpha = 0.05, coef_change_threshold = 0.1) {
  # Vérification que la variable réponse est bien un facteur
  if (!is.factor(data[[response_var]])) {
    data[[response_var]] <- as.factor(data[[response_var]])
  }
  
  # Formule initiale
  formula_current <- as.formula(paste(response_var, "~ ."))
  
  # Ajustement du modèle initial
  model_current <- vglm(formula_current, data = data, family = multinomial)
  
  # Boucle de sélection descendante
  repeat {
    # Test ANOVA pour identifier la variable la moins significative
    anova_results <- anova(model_current, type = "I", test = "Chisq")
    
    # Extraction des p-values (on ignore la première ligne qui correspond à l'intercept)
    if (nrow(anova_results) <= 1) break  # Plus de variables à supprimer
    
    p_values <- anova_results[-1, "Pr(>Chi)"]
    max_p_var <- names(which.max(p_values))
    
    # Si la p-value max est > alpha, on essaie de supprimer la variable
    if (max(p_values) > alpha) {
      # Modèle réduit
      formula_reduced <- update(formula_current, paste(". ~ . -", max_p_var))
      model_reduced <- vglm(formula_reduced, data = data, family = multinomial)
      
      # Test de différence de déviance
      deviance_diff <- deviance(model_reduced) - deviance(model_current)
      df_diff <- df.residual(model_reduced) - df.residual(model_current)
      p_value_deviance <- pchisq(abs(deviance_diff), df_diff, lower.tail = FALSE)
      
      # Vérification des changements dans les coefficients significatifs
      coef_current <- coef(model_current)
      coef_reduced <- coef(model_reduced)
      
      # On ne compare que les coefficients présents dans les deux modèles
      common_coefs <- intersect(names(coef_current), names(coef_reduced))
      coef_current <- coef_current[common_coefs]
      coef_reduced <- coef_reduced[common_coefs]
      
      # Identification des coefficients significatifs qui changent beaucoup
      for (coef_name in common_coefs) {
        # On vérifie d'abord si le coefficient est significatif dans le modèle courant
        summ <- summary(model_current)
        p_val_coef <- summ@coef3[coef_name, "Pr(>|z|)"]
        
        if (!is.na(p_val_coef) && p_val_coef < alpha) {
          change <- abs((coef_reduced[coef_name] - coef_current[coef_name]) / abs(coef_current[coef_name])
                        if (change > coef_change_threshold) {
                          warning(paste("Attention: Le coefficient de", coef_name, 
                                        "a changé de", round(change*100, 1), 
                                        "% après suppression de", max_p_var,
                                        "(ancienne valeur:", round(coef_current[coef_name], 4),
                                        "nouvelle valeur:", round(coef_reduced[coef_name], 4), ")"))
                        }
        }
      }
      
      # Si la réduction n'est pas significativement pire, on l'accepte
      if (p_value_deviance > alpha) {
        model_current <- model_reduced
        formula_current <- formula_reduced
        cat(paste("Variable", max_p_var, "supprimée (p-value ANOVA:", round(max(p_values), 4), 
                  "| p-value test déviance:", round(p_value_deviance, 4), ")\n")
      } else {
        cat(paste("Arrêt: la suppression de", max_p_var, 
                  "détériore significativement le modèle (p-value test déviance:", 
                  round(p_value_deviance, 4), ")\n")
            break
      }
    } else {
      cat("Arrêt: toutes les variables restantes sont significatives (p-value <", alpha, ")\n")
      break
    }
  }
  
  # Affichage du modèle final
  cat("\nModèle final:\n")
  print(formula_current)
  cat("\nRésumé du modèle final:\n")
  print(summary(model_current))
  
  return(model_current)
}
