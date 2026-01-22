# Music data analysis using Billboard Hot-100 and Spotify audio features
# This script cleans the data, builds predictive models, and creates visuals
# for the IJC437 and IJC445 coursework projects
#installations
install.packages(c("data.table", "tidyverse", "caret", "pROC", "randomForest"))
# Clear everything from the environment to start fresh
rm(list = ls())

# Load libraries used for data handling, modelling, and visualisation
library(data.table)
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tibble)

# Set the working directory where the dataset is stored
bb_raw <- fread("data/BillboardDataset.csv", encoding = "UTF-8")

# Keep only the columns needed for this analysis
bb <- bb_raw[, c(
  "song", "band_singer", "ranking", "year", "lyrics",
  "danceability", "energy", "loudness", "speechiness",
  "acousticness", "instrumentalness", "liveness",
  "valence", "tempo", "duration_ms"
), with = FALSE]

# Rename columns to make them easier to understand
bb <- bb %>%
  rename(
    artist = band_singer,
    rank   = ranking
  )

# List of Spotify audio features used throughout the analysis
audio_features <- c(
  "danceability", "energy", "loudness", "speechiness",
  "acousticness", "instrumentalness",
  "liveness", "valence", "tempo"
)

# Remove songs with missing audio feature values
bb <- bb %>%
  drop_na(all_of(audio_features))

# Some songs appear multiple times across different weeks
# These are collapsed into one record per song and artist
bb_clean <- bb %>%
  group_by(song, artist) %>%
  summarise(
    rank   = min(rank, na.rm = TRUE),
    year   = min(year, na.rm = TRUE),
    lyrics = first(lyrics),
    across(all_of(audio_features), mean),
    .groups = "drop"
  )

# Create a binary outcome variable indicating chart success
# Songs reaching the Top 10 are labelled as hits
bb_clean <- bb_clean %>%
  mutate(hit = if_else(rank <= 10, 1, 0))

# Apply simple filters to remove implausible values
bb_clean <- bb_clean %>%
  filter(
    tempo > 0,
    between(danceability, 0, 1),
    between(energy, 0, 1)
  )

# Scale audio features so they are comparable in the models
bb_clean <- bb_clean %>%
  mutate(across(all_of(audio_features), scale))

# Convert the target variable into a factor for classification
bb_clean$hit <- factor(
  bb_clean$hit,
  levels = c(0, 1),
  labels = c("NoHit", "Hit")
)

# Create the final dataset used for modelling
model_data <- bb_clean %>%
  select(hit, all_of(audio_features)) %>%
  droplevels()

# Set up cross-validation for model training
set.seed(123)

ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train a logistic regression model as a baseline
logit_model <- train(
  hit ~ .,
  data = model_data,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)

# Generate predicted probabilities and ROC curve
logit_prob <- predict(logit_model, model_data, type = "prob")
roc_logit <- roc(model_data$hit, logit_prob$Hit)

# Train a random forest model to capture non-linear relationships
rf_model <- train(
  hit ~ .,
  data = model_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 5
)

# Generate predicted probabilities and ROC curve
rf_prob <- predict(rf_model, model_data, type = "prob")
roc_rf <- roc(model_data$hit, rf_prob$Hit)

# Compare the two models using ROC curves
plot(roc_logit, col = "blue", lwd = 2,
     main = "ROC Curve Comparison")
plot(roc_rf, col = "red", lwd = 2, add = TRUE)

legend(
  "bottomright",
  legend = c(
    paste("Logistic AUC =", round(auc(roc_logit), 3)),
    paste("Random Forest AUC =", round(auc(roc_rf), 3))
  ),
  col = c("blue", "red"),
  lwd = 2,
  bty = "n"
)

# Visualise feature importance from the random forest model
rf_importance <- varImp(rf_model, scale = TRUE)
plot(rf_importance, top = 10,
     main = "Random Forest Feature Importance")

# Explore correlations between Spotify audio features
corr_long <- cor(model_data[, -1], use = "complete.obs") |>
  as.data.frame() |>
  rownames_to_column("Feature1") |>
  pivot_longer(-Feature1,
               names_to = "Feature2",
               values_to = "Correlation")

ord <- colnames(model_data[, -1])
corr_long$Feature1 <- factor(corr_long$Feature1, ord)
corr_long$Feature2 <- factor(corr_long$Feature2, ord)

ggplot(corr_long, aes(Feature1, Feature2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient2(
    low = "#457B9D", mid = "#F1FAEE", high = "#E76F51",
    midpoint = 0, limits = c(-1, 1)
  ) +
  coord_fixed() +
  labs(
    title = "Correlation Between Spotify Audio Features",
    subtitle = "Pairwise Pearson correlations of scaled audio features",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(color = "grey40", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

# Scatter plot showing the relationship between danceability and energy
# Colours distinguish hit and non-hit songs
hit_col   <- "#C8553D"
nohit_col <- "#2A9D8F"

cols <- ifelse(model_data$hit == "Hit", hit_col, nohit_col)

plot(
  model_data$danceability,
  model_data$energy,
  col = adjustcolor(cols, alpha.f = 0.65),
  pch = 16,
  cex = 1.2,
  xlab = "Danceability (scaled)",
  ylab = "Energy (scaled)",
  main = "Danceability vs Energy by Song Success"
)

grid(col = "grey88", lty = "dotted")

legend(
  "bottomright",
  legend = c("Hit", "NoHit"),
  col = c(hit_col, nohit_col),
  pch = 16,
  bty = "n"
)

# Track how the proportion of hit songs changes over time
hit_by_year <- aggregate(
  hit ~ year,
  data = bb_clean,
  FUN = function(x) mean(x == "Hit")
)

plot(
  hit_by_year$year,
  hit_by_year$hit,
  type = "b",
  pch = 16,
  lwd = 2,
  col = "#2A9D8F",
  xlab = "Year",
  ylab = "Proportion of Top-10 Hits",
  main = "Proportion of Hit Songs Over Time"
)

lines(
  lowess(hit_by_year$year, hit_by_year$hit),
  col = "#E76F51",
  lwd = 2
)

legend(
  "topleft",
  legend = c("Yearly proportion", "Smoothed trend"),
  col = c("#2A9D8F", "#E76F51"),
  lwd = 2,
  pch = c(16, NA),
  bty = "n"
)

grid(col = "grey88", lty = "dotted")

# Examine how speechiness has evolved across years
speech_by_year <- aggregate(
  speechiness ~ year,
  data = bb_clean,
  mean
)

plot(
  speech_by_year$year,
  speech_by_year$speechiness,
  type = "b",
  pch = 16,
  lwd = 2,
  col = "#5A4FCF",
  xlab = "Year",
  ylab = "Average Speechiness",
  main = "Average Speechiness of Songs Over Time"
)

lines(
  lowess(speech_by_year$year, speech_by_year$speechiness),
  col = "#E76F51",
  lwd = 2
)

legend(
  "topleft",
  legend = c("Yearly average", "Smoothed trend"),
  col = c("#5A4FCF", "#E76F51"),
  lwd = 2,
  pch = c(16, NA),
  bty = "n"
)

grid(col = "grey88", lty = "dotted")
 
