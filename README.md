**What Makes a Hit?**
**An Analysis of Musical Audio Features and Billboard Hot-100 Success (2000–2023)**
This project investigates whether measurable musical characteristics can explain or predict a song’s success on the Billboard Hot-100 chart. Using Spotify audio features combined with Billboard chart data from 2000 to 2023, the analysis explores how musical attributes relate to commercial popularity.

This work was completed as part of the IJC437 – Introduction to Data Science module and is intended for both an academic audience and a professional audience, including potential employers or clients interested in applied data science and artificial intelligence.

**Research Questions**

The study is guided by the following research questions, which are intentionally model-agnostic and focus on relationships rather than any single analytical technique.

1. To what extent can Spotify audio features predict whether a song reaches the Billboard Top-10?
2. Which Spotify audio features are most strongly associated with chart success?
3. How have musical characteristics, particularly speechiness, evolved over time in popular music?

**Data Description**

The analysis uses a single integrated dataset consisting of Billboard Hot-100 chart data from 2000 to 2023 and Spotify audio features. The audio features included in the analysis are danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo.

Songs that reached the Top-10 at least once are classified as Hit, while all other songs are classified as NoHit. This definition reflects a clear and widely recognised threshold of mainstream commercial success.

**Methods Used**

The project follows a structured data science workflow.
Data cleaning and preprocessing involved removing missing values, aggregating repeated chart entries to avoid bias from long-charting songs, and standardising audio features using z-score scaling.
Exploratory data analysis was conducted using correlation heatmaps, scatter plots, and time-series visualisations to understand relationships between features and how musical characteristics have changed over time.
Predictive modelling was carried out using logistic regression as an interpretable baseline model and random forest as a non-linear ensemble model. Both models were trained using 10-fold cross-validation and evaluated using ROC–AUC to account for class imbalance.

**Key Findings**

The results show that Spotify audio features alone have limited predictive ability when it comes to explaining commercial success on the Billboard Hot-100. The audio features most strongly associated with hit songs are speechiness, energy, and danceability.
Random forest models perform a slight improvement over logistic regression, indicating a notion of week nonlinear relationships between musical variables. However, predictive capacity remains low. A definite trend emerges with respect to increased usage in speech-oriented vocals, representing the growing popularity of rap and spoken word music in mainstream culture.
This shows that though musical features have significance, another important aspect is social exposure or marketing, apart from which an artist or song cannot become a chart-topper.


IJC437-Music-Hit-Analysis
data – dataset files
figures – generated plots and visualisations
scripts – fully commented R analysis script
report – final coursework report
README – project overview

How to Run the Code

First, clone the repository from GitHub.
Next, open the analysis.R file in RStudio.
Set the working directory to the project folder.
Install the required packages if they are not already installed.
Run the script from top to bottom to reproduce all results and figures.

**About Me**

I am a data science student with a keen interest in artificial intelligence and machine learning, and I am particularly enthusiastic about tackling real-world datasets with quantitative and artificial intelligence techniques. This particular project showcases my capability in processing and analyzing large data sets, developing statistical and machine learning models, and presenting results to technical and non-technical audiences alike.

**Notes**

This repository will remain unchanged after submission, in line with coursework requirements. All results are fully reproducible using the provided data and code.
