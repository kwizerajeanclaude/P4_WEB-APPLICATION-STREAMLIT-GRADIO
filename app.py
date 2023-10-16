#Importing the libraries
import gradio as gr
import pickle
import pandas as pd
import numpy as np
import joblib
from PIL import Image

#using joblib to load the model:
num_imputer = joblib.load('num_imputer.joblib') # loading the imputer 
cat_imputer = joblib.load('cat_imputer.joblib') # loading the imputer
encoder = joblib.load('encoder.joblib') # loading the encoder
scaler = joblib.load('scaler.joblib') # loading the scaler
model = joblib.load('ml.joblib') # loading the model


# Create a function that applies the ML pipeline and makes predictions
def predict(gender,SeniorCitizen,Partner,Dependents, tenure, PhoneService,MultipleLines,
                       InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                       Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):



    # Create a dataframe with the input data
     input_df = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]

 })

# Create a list with the categorical and numerical columns
     cat_columns = [col for col in input_df.columns if input_df[col].dtype == 'object']
     num_columns = [col for col in input_df.columns if input_df[col].dtype != 'object']

    # Impute the missing values
     input_df_imputed_cat = cat_imputer.transform(input_df[cat_columns]) 
     input_df_imputed_num = num_imputer.transform(input_df[num_columns]) 

    # Encode the categorical columns
     input_encoded_df = pd.DataFrame(encoder.transform(input_df_imputed_cat).toarray(),
                                   columns=encoder.get_feature_names_out(cat_columns))

    # Scale the numerical columns
     input_df_scaled = scaler.transform(input_df_imputed_num)
     input_scaled_df = pd.DataFrame(input_df_scaled , columns = num_columns)


    #joining the cat encoded and num scaled
     final_df = pd.concat([input_encoded_df, input_scaled_df], axis=1)

     final_df = final_df.reindex(columns=['SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
     'gender_Female','gender_Male','Partner_No','Partner_Yes','Dependents_No','Dependents_Yes','PhoneService_No',
     'PhoneService_Yes','MultipleLines_No','MultipleLines_Yes','InternetService_DSL','InternetService_Fiber optic',
     'InternetService_No','OnlineSecurity_No','OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_Yes','DeviceProtection_No',
     'DeviceProtection_Yes','TechSupport_No','TechSupport_Yes','StreamingTV_No','StreamingTV_Yes','StreamingMovies_No',
     'StreamingMovies_Yes','Contract_Month-to-month','Contract_One year','Contract_Two year','PaperlessBilling_No',
     'PaperlessBilling_Yes','PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
     'PaymentMethod_Mailed check'])

    # Make predictions using the model
     predict = model.predict(final_df)


     prediction_label = "THIS CUSTOMER WILL CHURN" if predict.item() == "Yes" else "THIS CUSTOMER WILL NOT CHURN"


     return prediction_label

     #return predictions

#define the input interface


input_interface = []

with gr.Blocks(css=".gradio-container {background-color:silver}") as app:
    title = gr.Label('CUSTOMER CHURN PREDICTION App')
    

 
    with gr.Row():
        gr.Markdown("This application provides predictions on whether a customer will churn or remain with the Company. Please enter the customer's information below and click PREDICT to view the prediction outcome.")

    with gr.Row():
        with gr.Column(scale=3.5, min_width=500):
            input_interface = [
                gr.components.Radio(['male', 'female'], label='What is your Gender?'),
                gr.components.Number(label="Are you a Seniorcitizen? (No=0 and Yes=1), 55years and above"),
                gr.components.Radio(['Yes', 'No'], label='Do you have a Partner?'),
                gr.components.Dropdown(['No', 'Yes'], label='Do you have any Dependents?'),
                gr.components.Number(label='Length of Tenure (No. of months with Vodafone)'),
                gr.components.Radio(['No', 'Yes'], label='Do you use Phone Service?'),
                gr.components.Radio(['No', 'Yes'], label='Do you use Multiple Lines?'),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='Do you use Internet Service?'),
                gr.components.Radio(['No', 'Yes'], label='Do you use Online Security?'),
                gr.components.Radio(['No', 'Yes'], label='Do you use Online Backup?'),
                gr.components.Radio(['No', 'Yes'], label='Do you use Device Protection?'),
                gr.components.Radio(['No', 'Yes'], label='Do you use the Tech Support?'),
                gr.components.Radio(['No', 'Yes'], label='Do you Streaming TV?'),
                gr.components.Radio(['No', 'Yes'], label='Do you Streaming Movies?'),
                gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='Please what Contract Type do you Subscribe to?'),
                gr.components.Radio(['Yes', 'No'], label='Do you use Paperless Billing?'),
                gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                        'Credit card (automatic)'], label='What type of Payment Method do you use please?'),
                gr.components.Number(label="How much is you Monthly Charges?"),
                gr.components.Number(label="How much is your Total Charges?")
            ]

    with gr.Row():
        predict_btn = gr.Button('Predict') 
        
 

# Define the output interfaces
    output_interface = gr.Label(label="churn", type="label", style="font-weight: bold; font-size: larger; color: red")

    predict_btn.click(fn=predict, inputs=input_interface, outputs=output_interface)


app.launch()
