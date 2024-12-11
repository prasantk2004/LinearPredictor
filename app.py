import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

st.set_page_config(
    page_title="Linear Regression App",
    page_icon="🧊",
    layout="wide"
)

st.sidebar.header("1. Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.dataframe(data, height=200)

    st.sidebar.header("2. Select Variables")
    independent_vars = st.sidebar.multiselect(
        "Select Independent Variable(s) (X)", options=data.columns.tolist()
    )
    dependent_var = st.sidebar.selectbox(
        "Select Dependent Variable (Y)", options=data.columns.tolist()
    )

    if independent_vars and dependent_var:
        X = data[independent_vars]
        y = data[dependent_var]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        coefficients = model.coef_
        intercept = model.intercept_
        equation = f"y = {intercept:.2f} "
        for i, coef in enumerate(coefficients):
            equation += f"+ ({coef:.2f} * {independent_vars[i]}) "

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y, y_pred)

        st.write("### Model Equation")
        st.latex(equation)

        st.write("### Performance Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"R² Score: {r_squared:.4f}")

        st.write("### Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=y, y=y_pred, ax=ax)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color="red", lw=2)
            ax.set_title("Observed vs Predicted")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

        with col2:
            residuals = y - y_pred
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Residuals Distribution")
            st.pyplot(fig)

        st.write("### Make Predictions")
        with st.expander("Input Custom Values for Prediction"):
            custom_input = st.text_area(
                "Enter comma-separated values for X (e.g., 5.1, 3.5)", ""
            )
            if custom_input:
                try:
                    custom_input = np.array(
                        [float(x.strip()) for x in custom_input.split(",")]
                    ).reshape(1, -1)
                    prediction = model.predict(custom_input)
                    st.write(f"Predicted Value: {prediction[0]:.4f}")
                except ValueError:
                    st.error(
                        "Invalid input. Please enter numeric values "
                        "separated by commas."
                    )

        st.sidebar.header("3. Download Results")
        generate_pdf = st.sidebar.checkbox("Generate Report as PDF")

        if generate_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Linear Regression Report", ln=True,
                     align="C")
            pdf.ln(10)

            pdf.cell(0, 10, f"Uploaded File: {uploaded_file.name}", ln=True)
            pdf.cell(0, 10, f"Model Equation: {equation}", ln=True)
            pdf.cell(0, 10, f"Mean Absolute Error (MAE): {mae:.4f}", ln=True)
            pdf.cell(0, 10, f"Mean Squared Error (MSE): {mse:.4f}", ln=True)
            pdf.cell(0, 10,
                     f"Root Mean Squared Error (RMSE): {rmse:.4f}", ln=True)
            pdf.cell(0, 10, f"R² Score: {r_squared:.4f}", ln=True)

            pdf_file = "regression_report.pdf"
            pdf.output(pdf_file)

            with open(pdf_file, "rb") as file:
                st.sidebar.download_button(
                    label="Download Report as PDF",
                    data=file,
                    file_name=pdf_file,
                    mime="application/pdf",
                )
else:
    st.info("Upload a CSV file to begin.")
