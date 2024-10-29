# Forecasting Experiments

This document outlines the forecasting accuracy experiments conducted for both spatial and point-based wind speed predictions. The objective is to compare the performance and accuracy of forecasting approaches across spatial grids, forecasting horizons, and individual locations within the dataset.

## Spatial Forecasts vs Point Forecasts accuracy

### Criteria for forecasting accuracy

Forecasting accuracy will be assessed based on the following criteria:

- RMSE (Root Mean Squared Error): Provides insights into the standard deviation of prediction errors, useful for more interpretable results in terms of actual wind speed units.
- Anomaly Correlation Coefficient (ACC): Measures the linear relationship between actual and predicted wind speed values, useful for identifying overall alignment.
- Spatial Consistency: Specific to spatial forecasts, evaluating the smoothness and consistency across grid cells for spatial accuracy.

### Model types to fit both cases

**Attention Transformer**: Will be used for both the point and spatial forecasts to compare the same model on similar areas to identify where the models differ. 

### Testing methodology\

**Locations**:
**Time horizons**:

