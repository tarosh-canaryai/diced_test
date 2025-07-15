import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import io
import json
import re

API_KEY = st.secrets.get("GEMINI_API_KEY")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key.")
        model = None
else:
    st.error("Gemini API Key not found. Please set the 'GEMINI_API_KEY' in your Streamlit secrets or directly in the script.")
    model = None

RISK_FRAMEWORK_PROMPT = """
The Unified Strategic Employee Risk Framework

Rule #1: The Critical Integrity Risk
Risk Level: SUPER CRITICAL
DESCRIPTION: This employee exhibits behaviors that represent a fundamental mismatch with core professional ethics. The issue is not one of skill or performance but of character and conduct. They are considered a significant liability, and their departure is predicted to be swift and involuntary due to non-negotiable behaviors like misconduct, job abandonment, or severe performance issues rooted in a lack of duty.
TRIGGER: Integrity < 25 OR Work Ethic/Duty < 25.
Why this Trigger Makes Sense: This is the framework's "fire alarm." An Integrity score below 25 signals a profound lack of honesty or ethical grounding. A Work Ethic/Duty score below 25 indicates a catastrophic failure to meet basic professional obligations. Either one of these is a terminal indicator that almost always leads to a rapid, for-cause separation.
Predicted Outcome: Involuntary (Misconduct, Abandoned Job, No-show, Performance)
Data-Driven Timeline: 0-6 Months

Rule #2: The Volatile High-Performer
Risk Level: CRITICAL
DESCRIPTION: This is the "brilliant jerk" or "high-maintenance superstar." They are exceptionally talented and deliver outstanding results, but they create organizational friction, challenge authority, and can be difficult to manage. Their high performance often makes them aware of their value, and they are a flight risk if they feel constrained, unappreciated, or receive a better offer.
TRIGGER: Score > 90 AND GYR = 'RED' AND Manipulative > 70.
Why this Trigger Makes Sense: The Score > 90 confirms their elite performance. The GYR = 'RED' shows they are in active conflict with their environment or manager. The Manipulative > 70 is the key differentiator; it reveals they use their high value to bend rules or navigate situations for personal gain, creating instability. This combination pinpoints a valuable but combustible asset.
Predicted Outcome: Voluntary (Resignation)
Data-Driven Timeline: 3-9 Months

Rule #3: The Dissonant Performer
Risk Level: CRITICAL
DESCRIPTION: This is a valuable employee who is clearly struggling. They are proven performers, but their current RED status indicates a significant problem—perhaps burnout, frustration with a project, or a poor relationship with their team or manager. They are at a crossroads, and without intervention, they are likely to either leave voluntarily or have their performance decline until it becomes an involuntary issue.
TRIGGER: Score > 80 AND GYR = 'RED'.
Why this Trigger Makes Sense: This trigger is designed to flag a valuable asset in distress. The Score > 80 establishes that this is an employee worth saving. The GYR = 'RED' is an unambiguous signal that an intervention is required. It captures good employees having a bad time, a critical group to focus on.
Predicted Outcome: Involuntary (Performance) or Voluntary (Resignation)
Data-Driven Timeline: 4-12 Months

Rule #4: The High-Value Flight Risk
Risk Level: CRITICAL
DESCRIPTION: This employee is a "silent flight risk." They are a top performer, not causing any trouble, and appear stable from the outside. However, they are mentally disengaged and holding back discretionary effort, likely while searching for their next opportunity. Because they are not a "problem," their risk is often overlooked until their resignation is submitted.
TRIGGER: Score > 85 AND GYR = 'GREEN' AND Withholding > 75.
Why this Trigger Makes Sense: The Score > 85 and GYR = 'GREEN' profile them as a top, stable-seeming employee. The Withholding > 75 is the critical hidden signal. It indicates a conscious decision to stop giving 100%, a classic sign of someone who has already "checked out" and is planning their exit.
Predicted Outcome: Voluntary (Resignation)
Data-Driven Timeline: 6-12 Months

Rule #5: The Direct Under-Performer
Risk Level: HIGH
DESCRIPTION: This employee is simply not meeting the basic expectations of the role. The issue is a clear and direct lack of performance and results, with little ambiguity. They are typically identified quickly and managed out through a formal performance improvement plan or direct termination.
TRIGGER: Score < 50 AND Achievement < 50.
Why this Trigger Makes Sense: This combination leaves no room for doubt. The low overall Score flags a general problem, and the low Achievement score specifically confirms that the employee is failing to produce results or meet goals. It's the most straightforward signal of performance-based termination.
Predicted Outcome: Involuntary (Performance)
Data-Driven Timeline: 0-6 Months

Rule #6: The Complacent Contributor
Risk Level: MEDIUM
DESCRIPTION: This employee is "coasting." They are not a problem employee and fly under the radar, but they are not striving, growing, or fully engaged. They represent a slow leak of potential and are at risk of leaving for a more compelling role or being selected for a reorganization because they are not seen as essential.
TRIGGER: GYR = 'GREEN' AND (Achievement < 60 OR Withholding > 60).
Why this Trigger Makes Sense: The GYR = 'GREEN' ensures this rule catches people who are not on a manager's immediate "problem" list. The OR condition is key: it finds people who are either not delivering results (Achievement < 60) or are actively disengaged (Withholding > 60), both of which define complacency.
Predicted Outcome: Voluntary (Stagnation) or Involuntary (Reorganization)
Data-Driven Timeline: 9-18 Months

Rule #7: The Ideal Core Employee
Risk Level: LOW
DESCRIPTION: This is the organizational bedrock. They are high-performing, highly engaged, trustworthy, and aligned with the company's goals and culture. They are the model employees you can build a team around and should be the focus of long-term retention and development efforts.
TRIGGER: Score > 85 AND GYR = 'GREEN' AND Integrity > 70 AND Withholding < 50 AND Manipulative < 50.
Why this Trigger Makes Sense: This is a rule of "positive confirmation with disqualifiers." It requires excellent performance (Score > 85) and alignment (GYR > 70) while explicitly filtering out anyone with even moderate integrity issues, disengagement, or manipulative tendencies. An employee must pass every one of these checks to earn this classification.
Predicted Outcome: Active (Stable)
Data-Driven Timeline: > 18 Months

Rule #8: The Stable Employee (Gray Zone)
Risk Level: LOW
DESCRIPTION: This employee is the default category. They do not exhibit any of the strong risk signals from the profiles above, but they also do not meet the elite criteria of an "Ideal Core Employee." They are considered generally stable and are not an immediate flight or performance risk.
TRIGGER: The employee did not trigger any of the rules from 1 through 7.
Predicted Outcome: Active (Stable)
Data-Driven Timeline: > 18 Months

Final Master Prompt for Automated Analysis
"Analyze the provided active employee data using the Unified Strategic Employee Risk Framework. The framework is hierarchical. For the given employee, perform the following steps in order. Assign them to the first risk profile they trigger and provide the specified outputs.
**Infer the employee's name from the provided data fields if a clear name column exists (e.g., 'Name', 'Employee Name', 'Full Name', 'First Name', 'Last Name'). If no discernible name is available, leave the 'Name' field in the output empty.**

The output should be a JSON object containing the 'Name', 'Row Number', 'Risk Profile Name', 'Risk Level', 'Predicted Outcome', and 'Data-Driven Timeline'.

Example of expected JSON output for an employee with a 'Name' column:
{{
    "Name": "John Doe",
    "Row Number": "1",
    "Risk Profile Name": "Volatile High-Performer",
    "Risk Level": "CRITICAL",
    "Predicted Outcome": "Voluntary (Resignation)",
    "Data-Driven Timeline": "3-9 Months"
}}

Example of expected JSON output for an employee without a 'Name' column (using row number):
{{
    "Name": "",
    "Row Number": "2",
    "Risk Profile Name": "Ideal Core Employee",
    "Risk Level": "LOW",
    "Predicted Outcome": "Active (Stable)",
    "Data-Driven Timeline": "> 18 Months"
}}
"""

st.set_page_config(page_title="Employee Risk Framework Analyzer", layout="wide")
st.title("Unified Strategic Employee Risk Framework (Version 12.0)")
st.write("Analyze employee risk profiles using the predefined framework and Google Gemini.")

if not API_KEY:
    st.warning("Please configure your Gemini API Key in your Streamlit secrets or directly in `app.py`.")

input_method = st.radio("Choose input method:", ("Upload a file", "Enter data manually"))

df = None 

if input_method == "Upload a file":
    uploaded_file = st.file_uploader("Upload employee data (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.success("File uploaded successfully!")
            st.subheader("Uploaded Data:")
            st.dataframe(df, height=300)

        except Exception as e:
            st.error(f"Error reading the file: {e}. Please ensure it's a valid CSV or Excel file with the correct format.")

elif input_method == "Enter data manually":
    st.subheader("Enter Employee Data Manually:")
    manual_data = {}
    name_input = st.text_input("Employee Name (optional)", "")
    manual_data['Name'] = name_input

    col1, col2, col3 = st.columns(3)

    with col1:
        manual_data['Score'] = st.number_input("Score (0-100)", min_value=0, max_value=100, value=75, key="manual_score")
        manual_data['Integrity'] = st.number_input("Integrity (0-100)", min_value=0, max_value=100, value=75, key="manual_integrity")
        manual_data['Achievement'] = st.number_input("Achievement (0-100)", min_value=0, max_value=100, value=75, key="manual_achievement")

    with col2:
        manual_data['GYR'] = st.selectbox("GYR (Green/Yellow/Red)", ('GREEN', 'YELLOW', 'RED'), key="manual_gyr")
        manual_data['Work Ethic/Duty'] = st.number_input("Work Ethic/Duty (0-100)", min_value=0, max_value=100, value=75, key="manual_work_ethic")
        manual_data['Withholding'] = st.number_input("Withholding (0-100)", min_value=0, max_value=100, value=50, key="manual_withholding")

    with col3:
        manual_data['Conscientious'] = st.number_input("Conscientious (0-100)", min_value=0, max_value=100, value=75, key="manual_conscientious")
        manual_data['Manipulative'] = st.number_input("Manipulative (0-100)", min_value=0, max_value=100, value=50, key="manual_manipulative")
        manual_data['Organized'] = st.number_input("Organized (0-100)", min_value=0, max_value=100, value=75, key="manual_organized")
        manual_data['Anchor Cherry Picking'] = st.number_input("Anchor Cherry Picking (0-100)", min_value=0, max_value=100, value=50, key="manual_anchor_cherry_picking")

    df = pd.DataFrame([manual_data])
    st.subheader("Manually Entered Data:")
    st.dataframe(df, height=100) 

if df is not None:
    framework_required_columns = ['Score', 'GYR', 'Conscientious', 'Achievement', 'Organized',
                                  'Integrity', 'Work Ethic/Duty', 'Withholding', 'Manipulative',
                                  'Anchor Cherry Picking']

    missing_columns = [col for col in framework_required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Error: The data is missing the following required columns for risk assessment: {', '.join(missing_columns)}")
        st.info("Please ensure your data (uploaded file or manual entry) has all the necessary columns as specified.")
    else:
        if st.button("Analyze Employee Risk"):
            if model is None:
                st.error("Gemini API is not configured. Please ensure your API key is set correctly.")
            else:
                st.subheader("Analysis Results:")

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
                        current_row_number = index + 1 if input_method == "Upload a file" else len(st.session_state.cached_manual_results_df) + 1 if input_method == "Enter data manually" else 1
                        employee_name_for_prompt = employee_data.get('Name', '') 

                        employee_data_string = "\n".join([f"{col_name}: '{col_value}'" for col_name, col_value in employee_data.items()])

                        employee_prompt = f"""
                        I need you to apply the provided Unified Strategic Employee Risk Framework (Version 12.0) to the following employee's data.
                        Here is the framework:
                        {RISK_FRAMEWORK_PROMPT}

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

if input_method == "Enter data manually" and 'cached_manual_results_df' in st.session_state and not st.session_state.cached_manual_results_df.empty:
    if st.button("Clear Cached Manual Results"):
        st.session_state.cached_manual_results_df = pd.DataFrame(columns=[
            'Name', 'Row Number', 'Risk Profile Name', 'Risk Level',
            'Predicted Outcome', 'Data-Driven Timeline'
        ])
        st.rerun()