# IJC445 – Music Data Analysis & Visualisation
# Dataset: Billboard Hot-100 with Spotify Audio Features

# 0. Clear workspace to avoid conflicts

rm(list = ls())
# 1. Load required libraries

library(data.table)   # Fast data loading
library(tidyverse)    # Data manipulation & plotting
library(caret)        # Machine learning framework
library(pROC)         # ROC & AUC analysis
library(randomForest) # Random Forest model


# 2. Load dataset

bb_raw <- fread("BillboardDataset.csv", encoding = "UTF-8")

# 3. Select relevant columns for analysis

bb <- bb_raw[, c(
  "song",
  "band_singer",
  "ranking",
  "year",
  "lyrics",
  "danceability",
  "energy",
  "loudness",
  "speechiness",
  "acousticness",
  "instrumentalness",
  "liveness",
  "valence",
  "tempo",
  "duration_ms"
), with = FALSE]

# 4. Rename columns for clarity

bb <- bb %>%
  rename(
    artist = band_singer,
    rank   = ranking
  )

# 5. Define Spotify audio feature variables

audio_features <- c(
  "danceability","energy","loudness","speechiness",
  "acousticness","instrumentalness",
  "liveness","valence","tempo"
)

# 6. Remove rows with missing audio feature values

bb <- bb %>%
  drop_na(all_of(audio_features))

# 7. Aggregate duplicate song entries
#    - Best (minimum) chart rank
#    - Earliest year
#    - Mean audio features

bb_clean <- bb %>%
  group_by(song, artist) %>%
  summarise(
    rank   = min(rank, na.rm = TRUE),
    year   = min(year, na.rm = TRUE),
    lyrics = first(lyrics),
    across(all_of(audio_features), mean),
    .groups = "drop"
  )

# 8. Create binary success variable
#    Hit = Top 10
#    NoHit = Outside Top 10

bb_clean <- bb_clean %>%
  mutate(hit = if_else(rank <= 10, "Hit", "NoHit"))

# 9. Standardise audio features

bb_clean <- bb_clean %>%
  mutate(across(all_of(audio_features), scale))

# 10. Create modelling dataset

model_data <- bb_clean[, c("hit", audio_features)]
# 11. Train predictive models

set.seed(123)

# 12. Cross-validation setup
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# 13.Logistic Regression model
logit_model <- train(
  hit ~ .,
  data = model_data,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)

# 14.Random Forest model
rf_model <- train(
  hit ~ .,
  data = model_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)

# 15. Model evaluation using ROC & AUC

roc_logit <- roc(
  model_data$hit,
  predict(logit_model, model_data, type = "prob")[, "Hit"]
)

roc_rf <- roc(
  model_data$hit,
  predict(rf_model, model_data, type = "prob")[, "Hit"]
)

# 16. Define colour palette

hit_col   <- "#E76F51"
nohit_col <- "#457B9D"

# FIGURE 1: Danceability vs Energy Scatter Plot

cols <- ifelse(model_data$hit == "Hit", hit_col, nohit_col)

plot(
  model_data$danceability,
  model_data$energy,
  col = cols,
  pch = 16,
  cex = 1.2,
  xlab = "Danceability (scaled)",
  ylab = "Energy (scaled)",
  main = "Figure 1: Danceability vs Energy by Song Success"
)

legend(
  "bottomright",
  legend = c("Hit", "NoHit"),
  col = c(hit_col, nohit_col),
  pch = 16,
  bty = "n"
)

# FIGURE 2: Random Forest Partial Dependence Plots

rf_vis_data <- bb_clean %>%
  select(hit, all_of(audio_features)) %>%
  mutate(across(all_of(audio_features), as.numeric)) %>%
  as.data.frame()

rf_vis_data$hit <- factor(rf_vis_data$hit, levels = c("NoHit", "Hit"))

rf_vis <- randomForest(
  hit ~ .,
  data = rf_vis_data,
  ntree = 500
)

par(
  mfrow = c(2, 3),
  mar = c(4, 4, 3, 1),
  oma = c(0, 0, 2, 0)
)

for (feat in c("acousticness", "danceability", "energy",
               "loudness", "tempo", "valence")) {
  
  grid_vals <- seq(
    min(rf_vis_data[[feat]]),
    max(rf_vis_data[[feat]]),
    length.out = 50
  )
  
  pd_vals <- sapply(grid_vals, function(v) {
    temp <- rf_vis_data
    temp[[feat]] <- v
    mean(predict(rf_vis, temp, type = "prob")[, "Hit"])
  })
  
  plot(
    grid_vals,
    pd_vals,
    type = "l",
    lwd = 2,
    col = "#1D3557",
    xlab = paste(feat, "(scaled)"),
    ylab = "Predicted Probability of Hit",
    main = paste("Effect of", feat),
    ylim = c(0, 1)
  )
  
  rug(rf_vis_data[[feat]])
}

mtext(
  "Figure 2: Random Forest Partial Dependence Plots",
  outer = TRUE,
  cex = 1.2,
  font = 2
)

par(mfrow = c(1, 1))

# FIGURE 3: ROC Curve Comparison

plot(
  roc_logit,
  col = "#8D99AE",
  lwd = 2,
  main = "Figure 3: ROC Curve Comparison"
)

plot(
  roc_rf,
  col = hit_col,
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1, lty = 2, col = "grey70")

legend(
  "bottomright",
  legend = c(
    paste("Logistic AUC =", round(auc(roc_logit), 2)),
    paste("Random Forest AUC =", round(auc(roc_rf), 2))
  ),
  col = c("#8D99AE", hit_col),
  lwd = 2,
  bty = "n"
)

# FIGURE 4: Tempo Cumulative Distribution (ECDF)

plot(
  ecdf(model_data$tempo[model_data$hit == "NoHit"]),
  col = nohit_col,
  lwd = 2,
  main = "Figure 4: Cumulative Distribution of Tempo by Song Success",
  xlab = "Tempo (scaled)",
  ylab = "Cumulative Probability"
)

lines(
  ecdf(model_data$tempo[model_data$hit == "Hit"]),
  col = hit_col,
  lwd = 2
)

legend(
  "bottomright",
  legend = c("NoHit", "Hit"),
  col = c(nohit_col, hit_col),
  lwd = 2,
  bty = "n"
)
