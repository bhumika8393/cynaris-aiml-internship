# Feature Engineering Observations

## Encoding

- LabelEncoder converts categories into numeric labels.
- OneHotEncoder creates separate binary columns for each category.
- OrdinalEncoder is useful when categories have a natural order.

## Scaling

- StandardScaler centers data around mean 0 and standard deviation 1.
- MinMaxScaler scales values between 0 and 1.
- RobustScaler is less affected by outliers.

## Feature Selection

SelectKBest identified the most important features for predicting Marks based on statistical scores.

## Conclusion

Feature engineering improves machine learning performance by preparing data in a suitable numerical format and selecting useful features.
