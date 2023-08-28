import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import xgboost as xgb
from PIL import Image
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
df_data_featur = pd.read_csv("D:\model_deploy_ems\df_data_1k.csv")

# selction of the x and y
X = df_data_featur.drop(columns=['motor_speed'])
Y = df_data_featur['motor_speed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Load the XGB Regressor model with the best parameters
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)

# Train the XGBoost model on the training data
model.fit(X_train, y_train)

# Function to label encode categorical features
def label_encode_features(data):
    # Encode 'comfort_level' feature
    comfort_level_mapping = {'Comfortable': 0, 'Neutral': 1, 'Uncomfortable': 2}
    data['comfort_level'] = data['comfort_level'].map(comfort_level_mapping)

    # Encode 'wind_speed_category' feature
    wind_speed_mapping = {'Very Low Wind': 0, 'Low Wind': 1, 'Low to Mild Wind': 2, 'Mild Wind': 3,
                          'Mild to Moderate Wind': 4, 'Moderate Wind': 5, 'Moderate to Breezy Wind': 6,
                          'Breezy Wind': 7, 'Breezy to Windy': 8, 'Windy': 9, 'Windy to Very Windy': 10,
                          'Very Windy': 11, 'Very Windy to Strong': 12}
    data['wind_speed_category'] = data['wind_speed_category'].map(wind_speed_mapping)

    return data

# Function to gather user input for each feature
def user_input_features():
    input_features = {}
    # Gather user input for numeric features
    for col in ['ambient', 'coolant', 'u_d', 'u_q', 'torque', 'i_d', 'i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding', 'profile_id']:
        input_features[col] = st.sidebar.number_input(f"Enter {col}:", value=df_data_featur[col].mean())

    # Label encode the 'comfort_level' feature and display label encoded values as options
    comfort_level_mapping = {0: 'Comfortable', 1: 'Neutral', 2: 'Uncomfortable'}
    comfort_level_encoded = st.sidebar.selectbox("Select Comfort Level:", list(comfort_level_mapping.values()), index=1)
    input_features['comfort_level'] = list(comfort_level_mapping.keys())[list(comfort_level_mapping.values()).index(comfort_level_encoded)]

    # Label encode the 'wind_speed_category' feature and display label encoded values as options
    wind_speed_mapping = {0: 'Very Low Wind', 1: 'Low Wind', 2: 'Low to Mild Wind', 3: 'Mild Wind', 4: 'Mild to Moderate Wind',
                          5: 'Moderate Wind', 6: 'Moderate to Breezy Wind', 7: 'Breezy Wind', 8: 'Breezy to Windy',
                          9: 'Windy', 10: 'Windy to Very Windy', 11: 'Very Windy', 12: 'Very Windy to Strong'}
    wind_speed_encoded = st.sidebar.selectbox("Select Wind Speed Category:", list(wind_speed_mapping.values()), index=1)
    input_features['wind_speed_category'] = list(wind_speed_mapping.keys())[list(wind_speed_mapping.values()).index(wind_speed_encoded)]

    # Gather user input for the features you created
    input_features['temperature_fahrenheit'] = st.sidebar.number_input("Enter Temperature (Fahrenheit):", value=df_data_featur['temperature_fahrenheit'].mean())
    input_features['apparent_temperature'] = st.sidebar.number_input("Enter Apparent Temperature:", value=df_data_featur['apparent_temperature'].mean())
    input_features['wind_chill'] = st.sidebar.number_input("Enter Wind Chill:", value=df_data_featur['wind_chill'].mean())

    return input_features

# Function to add a home button
def home_button():
    st.sidebar.button('Home')

# Function to handle file upload and make predictions
def predict_from_file():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        sample_data = pd.read_csv(uploaded_file)

        # Label encode categorical features
        sample_data_encoded = label_encode_features(sample_data)

        # Rearrange the columns to match the order of X_train
        sample_data_encoded = sample_data_encoded[X_train.columns]

        # Handle user input in the original categorical form
        comfort_level_mapping = {0: 'Comfortable', 1: 'Neutral', 2: 'Uncomfortable'}
        wind_speed_mapping = {0: 'Very Low Wind', 1: 'Low Wind', 2: 'Low to Mild Wind', 3: 'Mild Wind', 4: 'Mild to Moderate Wind',
                              5: 'Moderate Wind', 6: 'Moderate to Breezy Wind', 7: 'Breezy Wind', 8: 'Breezy to Windy',
                              9: 'Windy', 10: 'Windy to Very Windy', 11: 'Very Windy', 12: 'Very Windy to Strong'}

        if 'comfort_level' in sample_data.columns and isinstance(sample_data['comfort_level'][0], str):
            sample_data_encoded['comfort_level'] = sample_data['comfort_level'].map(comfort_level_mapping)

        if 'wind_speed_category' in sample_data.columns and isinstance(sample_data['wind_speed_category'][0], str):
            sample_data_encoded['wind_speed_category'] = sample_data['wind_speed_category'].map(wind_speed_mapping)

        # Make predictions using the model
        predictions = model.predict(sample_data_encoded)

        # Display the predictions with a small title
        st.subheader("Sample Predicted Results:")
        st.write(predictions)

## Function to add a background image





# Function to add project members button
def project_members_button():
    if st.sidebar.button('Project Members'):
        st.write("Project Members:")
        st.write("- Gugloth Vijay")
        st.write("- Manil")
        st.write("- Aswarya")
        st.write("- pankaj")



st.set_page_config(layout="wide")

# Function to add background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.postimg.cc/VvKMkX5d/motor-image-1.jpg");
             background-attachment: fixed;
             background-position: center;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


custom_css = """
<style>
.logo {
    position: absolute;
    top: -25px;
    right: -15px;
    width: 100px; /* Adjust the width as needed */
    height: auto; /* Maintain aspect ratio */
    z-index: 100; /* Ensure the logo is above other elements */
}
</style>
"""


# Function to add visualization options
def visualization_options(data_columns):
    visualization_option = st.sidebar.selectbox("Select Visualization:", ["None", "Bar Chart", "Histogram", "Scatter Plot","Box Plot"])
    
    if visualization_option != "None":
        selected_column = st.sidebar.selectbox("Select Column:", data_columns)
        return visualization_option, selected_column
    else:
        return visualization_option, None

# Function to display selected visualization
def display_visualization(visualization_option, selected_column, data):
    if visualization_option == "Bar Chart":
        st.subheader("Bar Chart")
        if selected_column is not None:
            # Create a bar chart using seaborn
            fig, ax = plt.subplots()
            sns.barplot(x=selected_column, y='motor_speed', data=data, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select a column for visualization.")
    elif visualization_option == "Histogram":
        st.subheader("Histogram")
        if selected_column is not None:
            # Create a histogram using seaborn
            fig, ax = plt.subplots()
            sns.histplot(data[selected_column], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select a column for visualization.")
    elif visualization_option == "Scatter Plot":
        st.subheader("Scatter Plot")
        if selected_column is not None:
            # Create a scatter plot using seaborn
            fig, ax = plt.subplots()
            sns.scatterplot(x=selected_column, y='motor_speed', data=data, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select a column for visualization.")
    elif visualization_option == "Box Plot":
        st.subheader("Box Plot")
        if selected_column is not None:
            # Create a box plot using seaborn
            fig, ax = plt.subplots()
            sns.boxplot(x=selected_column, data=data, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please select a column for visualization.")



# Function to visualize predicted vs actual motor speed values with a scatter plot and prediction line
def visualize_prediction_vs_actual(sample_data, predictions):
    # Get the required length of actual motor speed values
    actual_motor_speeds = df_data_featur.loc[sample_data.index, 'motor_speed']

    # Create a DataFrame for visualization
    visualization_data = pd.DataFrame({'Predicted': predictions, 'Actual': actual_motor_speeds})

    # Create a scatter plot with red color for predicted values
    fig, ax = plt.subplots()
    ax.scatter(visualization_data['Actual'], visualization_data['Predicted'], color='blue', label='Actual vs Predicted')
    ax.plot([visualization_data['Actual'].min(), visualization_data['Actual'].max()],
            [visualization_data['Actual'].min(), visualization_data['Actual'].max()], color='red', linewidth=2, label='Prediction Line')
    ax.set_xlabel('Actual Motor Speed')
    ax.set_ylabel('Predicted Motor Speed')
    ax.set_title('Predicted vs Actual Motor Speed')
    ax.legend()

    st.pyplot(fig)

# Function to handle file upload and make predictions
def predict_from_file():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        sample_data = pd.read_csv(uploaded_file)

        # Label encode categorical features
        sample_data_encoded = label_encode_features(sample_data)

        # Rearrange the columns to match the order of X_train
        sample_data_encoded = sample_data_encoded[X_train.columns]

        # Make predictions using the model
        predictions = model.predict(sample_data_encoded)

        # Create a DataFrame with sample data and predictions
        combined_data = sample_data_encoded.copy()
        combined_data['Predicted Motor Speed'] = predictions

        # Display the combined data in a table format
        st.subheader("Sample Data with Predicted Motor Speed")
        st.write(combined_data)


# Function to get a download link for a DataFrame as a CSV file
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predicted_data.csv">Download Predicted Data</a>'


# Streamlit app code
def main():
    add_bg_from_url()

    # Customize your title here
    custom_title = "<h1 style='color: white; text-align: center;'>Model Deployment: XGB Regressor</h1>"

    # Your logo image tag without alt attribute
    logo_html = '<img src="https://i.postimg.cc/1X6mNRh0/Excel-R-Logo.jpg" class="logo">'

    # Combine title and logo in a single st.markdown() call
    st.markdown(custom_css, unsafe_allow_html=True)  # Apply custom CSS
    st.markdown(custom_title + logo_html, unsafe_allow_html=True)

    # Add a home button
    home_button()

    # Add project members button
    project_members_button()

    # Get the list of data columns
    data_columns = df_data_featur.columns

    # Add visualization options
    visualization_option, selected_column = visualization_options(data_columns)

    # Display selected visualization
    if visualization_option != "None":
        display_visualization(visualization_option, selected_column, df_data_featur)

    st.sidebar.header('User Input Parameters')

    # Get user input
    input_data = user_input_features()

    # Create a DataFrame with user inputs
    input_df = pd.DataFrame([input_data])

    # Label encode categorical features
    input_df_encoded = label_encode_features(input_df)

    # Rearrange the columns to match the order of X_train
    input_df_encoded = input_df_encoded[X_train.columns]

    # Make predictions using the trained model
    predictions = model.predict(input_df_encoded)

    # Display the predictions
    st.write("Predicted Motor Speed:", predictions[0])

    # Call predict_from_file function
    predict_from_file()

if __name__ == "__main__":
    main()
