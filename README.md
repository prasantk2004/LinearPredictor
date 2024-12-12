# Sample Python/Streamlit

LinearPredictor is an interactive web application designed to help users build and explore simple linear regression models. Users can upload their own datasets, choose predictor and response variables, and train a linear regression model. The app allows users to make predictions on new data, visualize the regression line, and evaluate model performance through interactive plots.

---

## 🚀 Getting Started  

### Open Using Daytona  

1. **Install Daytona**: Follow the [Daytona installation guide](https://www.daytona.io/docs/installation/installation/).

2. **Create the Workspace**:

   ```bash  
   daytona create https://github.com/prasantk2004/LinearPredictor.git
   ```

3. **Start the Application**:

   ```bash  
   streamlit run app.py
   ```

---

## ✨ Features  

- **CSV File Upload**: Easily upload your dataset (in CSV format).
- **Variable Selection**: Select the predictor and response variables from your dataset.
- **Train Model**: Build a simple linear regression model based on the selected variables.
- **Prediction**: Input new values to predict the response variable.
- **Scatter Plot**: Visualize the relationship between the predictor and response variables with a regression line.
- **Residual Plot**: Evaluate model performance by viewing the residuals (errors) from the regression model.
- **Interactive**: All steps are interactive, making it easy for users to experiment and understand the linear regression process.
