##laptop_prices project

pacman::p_load(tidyverse,     # data manipulation + visualization
               tidymodels,    # machine learning framework
               randomForest,  # random forest model
               xgboost,       # boosting model
               skimr,         # quick summary
               janitor,       # clean column names
               corrplot       # correlation plots
)

library(readr)

laptop_prices <- read_csv("data/laptop_prices.csv")

colnames(laptop_prices)

#Explanatory Analysis

glimpse(laptop_prices)

head(laptop_prices)

str(laptop_prices)

any(is.na(laptop_prices)) #no na's in the data-set

summary(laptop_prices)

#Numerical features Analysis
ggplot(laptop_prices, aes(x = Price_euros)) +
  geom_histogram(bins = 40, fill = "skyblue", color = "purple") +
  labs(title = "Distribution of Laptop Prices", x = "Price (€)", y = "Count")

#Correlation
numeric_cols <- laptop_prices %>%
  select(Ram, Weight, Price_euros, PrimaryStorage, SecondaryStorage, CPU_freq)

cor(numeric_cols, use = "complete.obs")

#Categogirical Analysis
ggplot(laptop_prices, aes(x = Company, y = Price_euros)) +
  geom_boxplot(fill = "orange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Laptop Price by Brand") #confirming brand laptop by their prices

ggplot(laptop_prices, aes(x = TypeName, y = Price_euros)) +
  geom_boxplot(fill = "lightgreen") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Laptop Price by Type") #just confirming if laptop price is assigned to its purpose

#correlation of price vs ram
ggplot(laptop_prices, aes(x = Ram, y = Price_euros)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Price vs RAM")

#relationship between price $ primary storage
ggplot(laptop_prices, aes(x = PrimaryStorage, y = Price_euros)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Price vs Primary Storage (GB)")


ggplot(laptop_prices, aes(x = Ram, y = Price_euros, color = Company)) +
  geom_point(alpha = 0.6) +
  labs(title = "Price vs RAM by Brand")

#Special Analysis

## Touch Screen
ggplot(laptop_prices, aes(x = Touchscreen, y = Price_euros)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Price by Touchscreen Feature")

#GPU
laptop_prices %>%
filter(!is.na(GPU_company)) %>%
  ggplot(aes(x = GPU_company, y = Price_euros)) +
  geom_boxplot(fill = "pink") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Price by GPU Company")


#CORRELATION MAP
library(reshape2)

num_data <- laptop_prices %>% 
  select(Ram, Weight, Price_euros, PrimaryStorage, SecondaryStorage, CPU_freq)

cor_mat <- cor(num_data, use = "complete.obs")
melted_cor <- melt(cor_mat)


ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Correlation Heatmap")


# features and targets

laptop_ml <- laptop_prices %>%
  select(Price_euros, Ram, Weight, PrimaryStorage, SecondaryStorage,
         CPU_freq, Company, TypeName, GPU_company, Touchscreen) %>%
  mutate(
    Company = as.factor(Company),
    TypeName = as.factor(TypeName),
    GPU_company = as.factor(GPU_company),
    Touchscreen = as.factor(Touchscreen)
  )


# Train and Test splits
library(tidymodels)
set.seed(123)
data_split <- initial_split(laptop_ml, prop = 0.8)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Recipe for preprocecing
recipe_laptop <- recipe(Price_euros ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%   # convert categorical variables
  step_zv(all_predictors())                  # remove zero variance predictors

# RF Model
library(randomForest)

rf_model <- rand_forest(mtry = 5, trees = 500, min_n = 5) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_recipe(recipe_laptop) %>%
  add_model(rf_model)

set.seed(123)

rf_fit <- rf_workflow %>%
  fit(data = train_data)


#XGB  Model
library(xgboost)

xgb_model <- boost_tree(trees = 500, tree_depth = 6, learn_rate = 0.1, loss_reduction = 0.01) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_workflow <- workflow() %>%
  add_recipe(recipe_laptop) %>%
  add_model(xgb_model)

set.seed(123)
xgb_fit <- xgb_workflow %>%
  fit(data = train_data)

#Evaluate the Models
# Random Forest predictions
rf_preds <- predict(rf_fit, test_data) %>%
  bind_cols(test_data)

# XGBoost predictions
xgb_preds <- predict(xgb_fit, test_data) %>%
  bind_cols(test_data)

# RMSE and R-squared for RF
rf_preds %>%
  metrics(truth = Price_euros, estimate = .pred)

# RMSE and R-squared for XGB
xgb_preds %>%
  metrics(truth = Price_euros, estimate = .pred)


#Featuring

#Importance
library(vip)
rf_fit %>% extract_fit_parsnip() %>% vip(num_features = 10)
xgb_fit %>% extract_fit_parsnip() %>% vip(num_features = 10)


#Predicted vs Actual plot

ggplot(xgb_preds, aes(x = Price_euros, y = .pred)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "XGBoost: Predicted vs Actual Prices", x = "Actual Price (€)", y = "Predicted Price (€)")

#Save Model
dir.create("models")

saveRDS(xgb_fit, "models/xgb_laptop_model.rds")

