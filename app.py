import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import io
import json
import re
from utils import RISK_FRAMEWORK_PROMPT_AI, RISK_FRAMEWORK_DISPLAY_PROMPT

# API Key handling for deployment
# Ensure your Streamlit Cloud secret is named 'GEMINI_API_KEY'
API_KEY = st.secrets.get("API_KEY")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key in Streamlit secrets.")
        model = None
else:
    st.error("Gemini API Key not found. Please set the 'GEMINI_API_KEY' in your Streamlit secrets.")
    model = None

st.set_page_config(
    page_title="Employee Risk Framework Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Unified Strategic Employee Risk Framework")
st.write("Analyze employee risk profiles using the predefined framework.")

# Add navigation to the sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Risk Analyzer", "View Risk Profiles"])

if page == "Risk Analyzer":
    st.markdown("---")
    st.header("Employee Risk Analyzer")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Input")
        st.write("Provide employee data either by uploading a file or entering it manually.")

        input_method = st.radio("Choose input method:", ("Upload a file", "Enter data manually"), key="main_input_method")

        df = None

        if input_method == "Upload a file":
            uploaded_file = st.file_uploader("Upload employee data (CSV or Excel)", type=["csv", "xlsx"], key="main_uploader")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)

                    st.success("File uploaded successfully!")
                    st.subheader("Uploaded Data Preview:")
                    st.dataframe(df, height=200)

                except Exception as e:
                    st.error(f"Error reading the file: {e}. Please ensure it's a valid CSV or Excel file with the correct format.")

        elif input_method == "Enter data manually":
            st.subheader("Enter Employee Data Manually:")
            manual_data = {}
            name_input = st.text_input("Employee Name (optional)", "", key="manual_name_input")
            manual_data['Name'] = name_input

            manual_col1, manual_col2, manual_col3 = st.columns(3) # Arranged into 3 columns

            with manual_col1:
                manual_data['Score'] = st.number_input("Score (0-100)", min_value=0, max_value=100, value=75, key="manual_score")
                manual_data['GYR'] = st.selectbox("GYR (Green/Yellow/Red)", ('GREEN', 'YELLOW', 'RED'), key="manual_gyr")
                manual_data['Conscientious'] = st.number_input("Conscientious (0-100)", min_value=0, max_value=100, value=75, key="manual_conscientious")

            with manual_col2:
                manual_data['Achievement'] = st.number_input("Achievement (0-100)", min_value=0, max_value=100, value=75, key="manual_achievement")
                manual_data['Organized'] = st.number_input("Organized (0-100)", min_value=0, max_value=100, value=75, key="manual_organized")
                manual_data['Integrity'] = st.number_input("Integrity (0-100)", min_value=0, max_value=100, value=75, key="manual_integrity")
            
            with manual_col3:
                manual_data['Work Ethic/Duty'] = st.number_input("Work Ethic/Duty (0-100)", min_value=0, max_value=100, value=75, key="manual_work_ethic")
                manual_data['Withholding'] = st.number_input("Withholding (0-100)", min_value=0, max_value=100, value=50, key="manual_withholding")
                manual_data['Manipulative'] = st.number_input("Manipulative (0-100)", min_value=0, max_value=100, value=50, key="manual_manipulative")
                manual_data['Anchor Cherry Picking'] = st.number_input("Anchor Cherry Picking (0-100)", min_value=0, max_value=100, value=50, key="manual_anchor_cherry_picking")


            df = pd.DataFrame([manual_data])
            st.subheader("Manually Entered Data Preview:")
            st.dataframe(df, height=100)

    with col2:
        st.subheader("Analysis Results")
        if df is not None:
            framework_required_columns = ['Score', 'GYR', 'Conscientious', 'Achievement', 'Organized',
                                          'Integrity', 'Work Ethic/Duty', 'Withholding', 'Manipulative',
                                          'Anchor Cherry Picking']

            missing_columns = [col for col in framework_required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Error: The data is missing the following required columns for risk assessment: {', '.join(missing_columns)}")
                st.info("Please ensure your data (uploaded file or manual entry) has all the necessary columns as specified.")
            else:
                if st.button("Analyze Employee Risk", key="analyze_button"):
                    if model is None:
                        st.error("Gemini API is not configured. Please ensure your API key is set correctly.")
                    else:
                        if input_method == "Enter data manually":
                            if 'cached_manual_results_df' not in st.session_state:
                                st.session_state.cached_manual_results_df = pd.DataFrame(columns=[
                                    'Name', 'Row Number', 'Risk Profile Name', 'Risk Level',
                                    'Predicted Outcome', 'Data-Driven Timeline'
                                ])
                            results_df = st.session_state.cached_manual_results_df
                        else:
                            results_df = pd.DataFrame(columns=[
                                'Name', 'Row Number', 'Risk Profile Name', 'Risk Level',
                                'Predicted Outcome', 'Data-Driven Timeline'
                            ])

                        results_container = st.container()
                        results_placeholder = results_container.dataframe(results_df, height=300)

                        with st.spinner("Analyzing employee data... This might take a while for large files."):
                            for index, row in df.iterrows():
                                employee_data = row.to_dict()
                                current_row_number = index + 1 if input_method == "Upload a file" else len(results_df) + 1
                                employee_name_for_prompt = employee_data.get('Name', '')

                                employee_data_string = "\n".join([f"{col_name}: '{col_value}'" for col_name, col_value in employee_data.items()])

                                employee_prompt = f"""
                                I need you to apply the provided Unified Strategic Employee Risk Framework (Version 12.0) to the following employee's data.
                                Here is the framework:
                                {RISK_FRAMEWORK_PROMPT_AI}

                                Here is the employee's data:
                                Row Number: {current_row_number}
                                {employee_data_string}

                                Based on the rules provided in the framework, determine the FIRST matching Risk Profile for this employee.
                                Your output MUST be a JSON object with 'Name', 'Row Number', 'Risk Profile Name', 'Risk Level', 'Predicted Outcome', and 'Data-Driven Timeline' as keys.
                                Example of expected JSON output:
                                {{
                                    "Name": "Employee Name If Found",
                                    "Row Number": "{current_row_number}",
                                    "Risk Profile Name": "Volatile High-Performer",
                                    "Risk Level": "CRITICAL",
                                    "Predicted Outcome": "Voluntary (Resignation)",
                                    "Data-Driven Timeline": "3-9 Months"
                                }}
                                If no name is discernible from the provided data, the 'Name' field should be an empty string, like: "Name": "".
                                """
                                try:
                                    response = model.generate_content(employee_prompt)

                                    response_text = response.text.strip()
                                    if response_text.startswith("```json") and response_text.endswith("```"):
                                        response_text = response_text[len("```json"):].rstrip("```").strip()

                                    employee_risk = json.loads(response_text)

                                    if 'Name' not in employee_risk or employee_risk['Name'] is None:
                                        employee_risk['Name'] = ""
                                    else:
                                        employee_risk['Name'] = str(employee_risk['Name'])

                                    employee_risk['Row Number'] = str(employee_risk.get('Row Number', current_row_number))

                                    new_row_df = pd.DataFrame([employee_risk])
                                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                                    results_placeholder.dataframe(results_df, height=300)

                                    if input_method == "Enter data manually":
                                        st.session_state.cached_manual_results_df = results_df

                                except json.JSONDecodeError as jde:
                                    st.warning(f"Could not parse JSON for row '{current_row_number}'. Error: {jde}. Raw response: {response.text}")
                                    failed_row = {
                                        "Name": employee_name_for_prompt,
                                        "Row Number": str(current_row_number),
                                        "Risk Profile Name": "Analysis Failed (JSON Error)",
                                        "Risk Level": "N/A",
                                        "Predicted Outcome": "N/A",
                                        "Data-Driven Timeline": "N/A"
                                    }
                                    new_row_df = pd.DataFrame([failed_row])
                                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                                    results_placeholder.dataframe(results_df, height=300)
                                    if input_method == "Enter data manually":
                                        st.session_state.cached_manual_results_df = results_df


                                except Exception as e:
                                    st.warning(f"Could not analyze row '{current_row_number}'. Error: {e}")
                                    failed_row = {
                                        "Name": employee_name_for_prompt,
                                        "Row Number": str(current_row_number),
                                        "Risk Profile Name": "Analysis Failed",
                                        "Risk Level": "N/A",
                                        "Predicted Outcome": "N/A",
                                        "Data-Driven Timeline": "N/A"
                                    }
                                    new_row_df = pd.DataFrame([failed_row])
                                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                                    results_placeholder.dataframe(results_df, height=300)
                                    if input_method == "Enter data manually":
                                        st.session_state.cached_manual_results_df = results_df

                        st.success("Analysis complete!")
                        if not results_df.empty:
                            csv_output = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv_output,
                                file_name="employee_risk_analysis_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No analysis results to display.")

                # Clear cache button moved to col2, only visible for manual input
                if input_method == "Enter data manually" and 'cached_manual_results_df' in st.session_state and not st.session_state.cached_manual_results_df.empty:
                    if st.button("Clear Cached Manual Results", key="clear_manual_cache_button_main"):
                        st.session_state.cached_manual_results_df = pd.DataFrame(columns=[
                            'Name', 'Row Number', 'Risk Profile Name', 'Risk Level',
                            'Predicted Outcome', 'Data-Driven Timeline'
                        ])
                        st.rerun()

elif page == "View Risk Profiles":
    st.markdown("---")
    st.header("Understanding Employee Risk Profiles")
    st.write("This section describes each risk profile within the Unified Strategic Employee Risk Framework.")
    st.markdown(RISK_FRAMEWORK_DISPLAY_PROMPT)
