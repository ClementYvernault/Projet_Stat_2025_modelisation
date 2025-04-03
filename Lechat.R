# Chargement des packages nécessaires
library(VGAM)
library(MASS)
library(car)
library(caret)
library(ggplot2)
library(reshape2)
library(pROC)

# Chargement des données
donnees_mean <- readRDS("data_post_etape_4_Mean.Rdata", "rb")
donnees_mean2 <- donnees_mean[,-1]

# Référence de la modalité "Sain"
donnees_mean2$y <- relevel(donnees_mean2$y, ref = "ENG_malade")

# Modélisation initiale avec toutes les variables
multinomial_logistic_model <- vglm(y ~ ., data = donnees_mean2, family = multinomial)

# Sélection pas à pas descendante
stepwise_model <- stepAIC(multinomial_logistic_model, direction = "backward")
stepwise_model <- modele_reduit3
# Résumé du modèle final
summary(stepwise_model)

# Calcul des log-vraisemblance, AIC, BIC
logLik(stepwise_model)
AIC(stepwise_model)
BIC(stepwise_model)

# Calcul du pseudo R²
pseudo_r2 <- 1 - (logLik(stepwise_model) / logLik(vglm(y ~ 1, data = donnees_mean2, family = multinomial)))
cat("Pseudo R² :", pseudo_r2, "\n")

# Vérification du VIF pour la multicolinéarité
vif_values <- vif(stepwise_model)
cat("VIF values :", vif_values, "\n")

# Prédictions et matrice de confusion
predictions <- predict(stepwise_model, newdata = donnees_mean2, type = "response")
classes_pred <- apply(predictions, 1, which.max)
classes_pred <- factor(classes_pred, levels = c(1, 2, 3), labels = c("ENG_malade", "PS_malade", "Sain"))

confusion.matrix <- table(True = donnees_mean2$y, Predicted = classes_pred)
confusion.matrix_norm <- prop.table(confusion.matrix, 1) * 100
round(confusion.matrix_norm, 1)

# Visualisation de la matrice de confusion
confusion_df <- melt(confusion.matrix_norm)
colnames(confusion_df) <- c("True", "Predicted", "Value")

ggplot(data = confusion_df, aes(x = True, y = Predicted, fill = Value)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Value)), color = "white") +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  labs(title = "Matrice de Confusion Normalisée", x = "Classe Réelle", y = "Classe Prédite")

# Précision et sensibilité
conf_matrix <- confusionMatrix(factor(classes_pred, levels = c("ENG_malade", "PS_malade", "Sain")), factor(donnees_mean2$y))
precision <- conf_matrix$byClass[,"Pos Pred Value"]
recall <- conf_matrix$byClass[,"Sensitivity"]

metrics_df <- data.frame(
  Classe = c("ENG_malade", "PS_malade", "Sain"),
  Precision = precision,
  Sensitivity = recall
)

metrics_df

# Courbes ROC et AUC
roc_list <- lapply(levels(donnees_mean2$y), function(level) {
  roc(donnees_mean2$y == level, predictions[, level])
})

# Tracer les courbes ROC
plot.roc(roc_list[[1]], col = "blue", print.auc = TRUE)
for (i in 2:length(roc_list)) {
  plot.roc(roc_list[[3]], add = FALSE, col = c("red", "green")[3-1], print.auc = TRUE)
}

# Validation croisée
control <- trainControl(method = "cv", number = 10)
cv_model <- train(y ~ ., data = donnees_mean2, method = "multinom", trControl = control)
print(cv_model)
