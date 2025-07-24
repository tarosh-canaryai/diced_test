import streamlit as st
import pandas as pd
import io
import os
import google.generativeai as genai
import re

st.set_page_config(layout="wide", page_title="Attrition & Hiring Risk Analyzer")

st.title("Model Comparison UI")

DECISION_TREE_RULEBOOK ="""
This is a hierarchical fall-through model. You must check an employee against the categories in order, from 7 down to 0. The first category an employee qualifies for is their definitive classification.
Category 7: The Volatile Performer
Risk Level: Highest Risk
Data Criteria: Manipulative > 95
Profile Description: An individual whose extreme score indicates a critical risk of creating a toxic environment, engaging in disruptive workplace politics, or being fundamentally unmanageable.
Estimated Tenure if Terminated: 1 - 5 Months
Actionable Insight: An unambiguous "Do Not Hire" signal. This is the most potent predictor of severe cultural damage and a rapid, chaotic exit.
Category 6: The Mismatch
Risk Level: Critical
Data Criteria: Conscientious < 50 AND Organized < 40 AND Integrity < 40
Profile Description: An individual whose scores indicate a fundamental inability to meet the basic requirements of the job. They are highly likely to struggle with reliability, organization, and following rules.
Estimated Tenure if Terminated: 0 - 3 Months
Actionable Insight: "Do Not Hire." This candidate is not set up for success, and the investment in training is highly likely to be lost.
Category 5: The High-Friction Employee
Risk Level: High
Data Criteria: Integrity < 50 AND Manipulative is between 70 and 95
Profile Description: A profile that flags a significant interpersonal risk. This employee may be productive but can erode team trust and cohesion over time through divisive or political behavior.
Estimated Tenure if Terminated: 2 - 7 Months
Actionable Insight: "Avoid Hiring." The risk of long-term damage to team morale outweighs short-term productivity.
Category 4: The Burnout Risk
Risk Level: Elevated
Data Criteria: (Conscientious > 80 AND Achievement > 90) AND (Work Ethic/Duty < 15)
Profile Description: The "sprinter, not a marathon runner." This individual is ambitious and capable of high performance but is at high risk of neglecting essential, routine duties, leading to rapid disengagement.
Estimated Tenure if Terminated: 1 - 4 Months
Actionable Insight: "Cautious Hire / Requires Strong Management." The combination of high ambition and very low diligence creates a highly volatile profile prone to a fast flameout.
Category 3: The Questionable Hire
Risk Level: Moderate-High
Data Criteria: Integrity < 60
Profile Description: This individual shows a significant weakness in the foundational area of integrity. This is a serious character flag that, while not as overt as higher-risk profiles, suggests a potential for untrustworthy behavior.
Estimated Tenure if Terminated: 2 - 6 Months
Actionable Insight: "Cautious Hire." Requires a role with high supervision and low autonomy. The risk of an integrity-related incident is a significant concern.
Category 2: The Disengaged Professional
Risk Level: Moderate
Data Criteria: (Conscientious > 80 AND Achievement > 90) AND (Work Ethic/Duty is between 15 and 40)
Profile Description: This is a less severe version of the "Burnout Risk." The individual is skilled and professional but has a notable lack of day-to-day drive. They are likely to "coast," doing the minimum required and posing a long-term flight risk if not actively engaged.
Estimated Tenure if Terminated: 6 - 12 Months
Actionable Insight: "Hire with a Plan." A viable candidate, but the hiring manager must be prepared with a clear plan for engagement, growth, and challenging assignments to maintain their interest.
Category 1: The Apathetic Hire
Risk Level: Moderate-Low
Data Criteria: Work Ethic/Duty < 40
Profile Description: This individual's primary flaw is a lack of drive or motivation. They do not have the critical behavioral or integrity issues of other profiles, but their low work ethic makes them an unpredictable and often inconsistent performer.
Estimated Tenure if Terminated: 3 - 7 Months
Actionable Insight: "Hire for simple, well-supervised roles only." Their success is highly dependent on the direct manager's ability to provide constant structure and motivation.
Category 0: The Steady Performer
Risk Level: Low
Data Criteria: The employee does not meet the criteria for Categories 1 through 7.
Profile Description: The target hiring profile. Demonstrates capability, a sound work ethic, and low behavioral risk. They are the foundation of a stable, productive team.
Estimated Tenure if Terminated: N/A (This profile predicts retention, not termination).
Actionable Insight: "Confident Hire." This is your priority candidate pool for stable, long-term success.
"""

if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_modified' not in st.session_state:
    st.session_state.df_modified = None
if 'current_row_start_index' not in st.session_state:
    st.session_state.current_row_start_index = 0
if 'gemini_per_row_results' not in st.session_state:
    st.session_state.gemini_per_row_results = []
if 'gemini_overall_report' not in st.session_state:
    st.session_state.gemini_overall_report = ""

def get_gemini_model():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except KeyError:
        st.error("Gemini API Key not found in Streamlit secrets. Please add it to your `secrets.toml` file.")
        return None
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key.")
        return None

def process_csv_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()
            st.session_state.df_modified = df.copy()
            st.session_state.current_row_start_index = 0
            st.session_state.gemini_per_row_results = []
            st.session_state.gemini_overall_report = ""
            st.success("CSV file uploaded successfully!")
            st.dataframe(st.session_state.df_modified.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

def get_rows_for_model(df, start_index, num_rows):
    if df is None:
        return pd.DataFrame()
    end_index = min(start_index + num_rows, len(df))
    return df.iloc[start_index:end_index]

def annotate_dataframe(df, percentage, columns_to_annotate):
    if df is None:
        st.warning("No CSV loaded to annotate.")
        return None

    df_copy = df.copy()
    for col in columns_to_annotate:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col] * (1 + percentage / 100)
        else:
            st.warning(f"Column '{col}' is not numeric or does not exist. Skipping annotation for it.")
    return df_copy

@st.cache_data
def convert_df_to_csv(df):
    if df is None:
        return ""
    return df.to_csv(index=False).encode('utf-8')

def clear_gemini_results():
    st.session_state.gemini_per_row_results = []
    st.session_state.gemini_overall_report = ""

try:
    _ = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.warning("Gemini API Key not found in Streamlit secrets. Please add it to your `secrets.toml` file to enable model functionality.")

st.header("Upload Your CSV Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None and st.session_state.df_original is None:
    process_csv_upload(uploaded_file)
elif uploaded_file is not None and uploaded_file.name != getattr(st.session_state.get('last_uploaded_filename'), 'name', ''):
    st.session_state.last_uploaded_filename = uploaded_file
    process_csv_upload(uploaded_file)

if st.session_state.df_modified is not None:
    st.subheader("Currently Loaded Data (first 5 rows):")
    st.dataframe(st.session_state.df_modified.head())

    st.header("Select Rows for Model Processing")
    max_rows = len(st.session_state.df_modified)
    num_rows_to_send = st.number_input(
        "Number of rows to send to model at a time (X):",
        min_value=1,
        max_value=max_rows if max_rows > 0 else 1,
        value=min(10, max_rows) if max_rows > 0 else 1,
        step=1
    )

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Next X Rows"):
            if st.session_state.current_row_start_index + num_rows_to_send < max_rows:
                st.session_state.current_row_start_index += num_rows_to_send
            else:
                st.session_state.current_row_start_index = 0
                st.info("Reached end of data. Resetting to start.")
            clear_gemini_results()
            st.rerun()
    with col_nav2:
        if st.button("Reset Rows Selection"):
            st.session_state.current_row_start_index = 0
            clear_gemini_results()
            st.rerun()

    st.info(f"Currently processing rows: {st.session_state.current_row_start_index} to "
            f"{min(st.session_state.current_row_start_index + num_rows_to_send, max_rows) - 1}")

    current_rows_df = get_rows_for_model(st.session_state.df_modified,
                                         st.session_state.current_row_start_index,
                                         num_rows_to_send)

    st.subheader("Rows being sent to models:")
    st.dataframe(current_rows_df)

    st.header("Annotate Numerical Values in CSV")
    if st.session_state.df_modified is not None:
        numerical_cols = st.session_state.df_modified.select_dtypes(include=['number']).columns.tolist()

        if numerical_cols:
            annotation_percentage = st.slider(
                "Select percentage change:",
                min_value=-50, max_value=50, value=0, step=1,
                help="Enter a positive value to increase, negative to decrease. E.g., 10 for +10%, -5 for -5%."
            )
            columns_to_annotate = st.multiselect(
                "Select column(s) to annotate:",
                options=numerical_cols,
                help="Choose one or more numerical columns to apply the percentage change."
            )

            if st.button("Annotate CSV"):
                if columns_to_annotate:
                    st.session_state.df_modified = annotate_dataframe(
                        st.session_state.df_modified,
                        annotation_percentage,
                        columns_to_annotate
                    )
                    st.success(f"CSV annotated! {annotation_percentage}% applied to {', '.join(columns_to_annotate)}.")
                    st.dataframe(st.session_state.df_modified.head())
                    st.session_state.current_row_start_index = 0
                    clear_gemini_results()
                else:
                    st.warning("Please select at least one numerical column to annotate.")
        else:
            st.info("No numerical columns found in the CSV to annotate.")

    st.header("Model Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gemini Decision Tree Model")
        try:
            _ = st.secrets["GEMINI_API_KEY"]
            if not current_rows_df.empty:
                model = get_gemini_model()
                if model:
                    if st.button("Run Gemini Model on Selected Rows"):
                        with st.spinner("Preparing to analyze..."):
                            per_row_summary_for_report = []
                            st.session_state.gemini_per_row_results = []

                            total_rows_to_process = len(current_rows_df)
                            status_placeholder = st.empty()

                            for idx, (_, row) in enumerate(current_rows_df.iterrows()):
                                current_processing_index = idx + 1
                                status_placeholder.info(f"Analyzing row {current_processing_index} of {total_rows_to_process}...")

                                row_csv_string = pd.DataFrame([row]).to_csv(index=False, header=True)

                                per_row_instruction_prompt = f"""
Using the "Definitive Attrition & Hiring Risk Model" rulebook provided below,
analyze each row of the following CSV data.
For each row, determine the exact "category" (e.g., "category 4: The Liability", "category 0a: The Cornerstone (Standard)")
that the employee falls into, applying the rules hierarchically from Category 4 down to 0a.
Then, provide a concise, one-line explanation *why* that specific category was assigned,
referencing the data criteria that led to that classification.

--- Definitive Attrition & Hiring Risk Model (Rulebook) ---
{DECISION_TREE_RULEBOOK}
--- End of Rulebook ---

--- Employee Data (CSV) ---
{row_csv_string}
--- End of Employee Data ---

Please format your output for each row as:
"Row {row.name}: [Category Name]. [One-line explanation]."
Example: "Row 1: category 4: The Liability. Score was 20 which is less than 25."
Example: "Row 2: category 0a: The Cornerstone (Standard). Met all criteria for 0a and did not meet any higher category criteria."
Ensure each row's analysis is on a new line.
You dont have to stick to the rules to explain the category that someone got, you may look at the other attributes if they explain the category better.
"""
                                try:
                                    response = model.generate_content(per_row_instruction_prompt)
                                    row_analysis = response.text.strip()
                                    st.session_state.gemini_per_row_results.append(row_analysis)
                                    per_row_summary_for_report.append(row_analysis)

                                except Exception as e:
                                    st.error(f"Error processing row {row.name} with Gemini: {e}")
                                    st.session_state.gemini_per_row_results.append(f"Error for Row {row.name}: {e}")
                                    per_row_summary_for_report.append(f"Error for Row {row.name}")

                            status_placeholder.empty()

                            st.subheader("Per-Row Analysis Details:")
                            with st.container(height=300, border=True):
                                if st.session_state.gemini_per_row_results:
                                    for idx, result_text in enumerate(st.session_state.gemini_per_row_results):
                                        display_row_index = st.session_state.current_row_start_index + idx

                                        display_content = result_text
                                        match = re.match(r"Row \d+:\s*(.*)", display_content, re.IGNORECASE)
                                        if match:
                                            display_content = match.group(1).strip()
                                        
                                        expander_title_suffix = "Analysis"
                                        category_match = re.match(r"(category \d+:\s*[^.]+)\.", display_content, re.IGNORECASE)
                                        if category_match:
                                            expander_title_suffix = category_match.group(1).strip()
                                        
                                        with st.expander(f"Row {display_row_index}: {expander_title_suffix}"):
                                            st.markdown(display_content)
                                else:
                                    st.info("No per-row results to display yet. Run the model to see results here.")


                            if per_row_summary_for_report:
                                st.subheader("Aggregate Summary Report:")
                                status_placeholder.info("Generating comprehensive aggregate report...")
                                overall_report_instruction_prompt = f"""
You have just classified a set of employee data rows into risk categories
using the "Definitive Attrition & Hiring Risk Model" rulebook.

Here is the immutable rulebook you used:
--- Definitive Attrition & Hiring Risk Model (Rulebook) ---
{DECISION_TREE_RULEBOOK}
--- End of Rulebook ---

Here are the individual classification results for each employee/row:
--- Per-Row Classification Results ---
{per_row_summary_for_report}
--- End of Per-Row Classification Results ---

Based on the rulebook and these individual classification results,
generate a comprehensive analysis report. Your report should be well-structured, clear,
and easy to understand for a hiring manager or HR professional.

**Structure your report precisely as follows, filling in the content based on your analysis:**

**Definitive Attrition & Hiring Risk Model: Analysis Report**
This report summarizes the results of applying the "Definitive Attrition & Hiring Risk Model" to a dataset of employee data. It highlights key trends, potential risks, and actionable insights to inform hiring and retention strategies.

**1. Risk Category Distribution:**
Summarize the count of employees in each risk category found in the "Per-Row Classification Results." Present this as a markdown table with columns for "Category," "Description," and "Count." Ensure all categories mentioned in the rulebook are included, even if their count is zero.

**2. Prevalent Risk Categories and Implications:**
Identify the 1-3 most prevalent risk categories based on their counts. For each identified category, provide:
- The **Category Name** and its **count**.
- Its **Profile Description** as stated in the rulebook.
- Its **Actionable Insight** as stated in the rulebook.
- A concise discussion of the **Implication** for hiring/HR based on the prevalence of this category within the analyzed dataset.

**3. Interesting Patterns and Anomalies:**
Highlight any noteworthy patterns or unusual observations from the classification results. This can include:
- Categories that have surprisingly high or low (including zero) counts.
- Any instances where employees narrowly missed a different category based on their scores.
- Specific data criteria (e.g., 'Withholding', 'Score', 'GYR') that appear to be strong drivers for certain classifications in this dataset.
- Mention any rows that stand out as exceptions or confirm specific model behaviors.

**4. Actionable Insights and Recommendations:**
Provide clear, specific, and actionable recommendations for a hiring manager or HR professional. These recommendations should directly stem from your analysis of the risk category distribution, prevalent risks, and observed patterns. Link recommendations to the "Actionable Insight" sections from the rulebook where appropriate.

**5. Impact of Data Changes (Causal Analysis):**
Discuss how hypothetical changes to the input data for *any* employee would causally impact their classification across risk categories. For each mentioned metric (e.g., 'Score', 'Manipulative', 'GYR', 'Work Ethic/Duty', 'Integrity'):
- Explain how increasing/decreasing its value could shift an employee from one category to another.
- Provide concrete, concise examples that directly reference the rulebook's criteria for category transitions.

**6. Conclusion:**
Provide a concise concluding summary of the report's main findings, emphasizing the most critical takeaways and strategic implications for managing attrition and hiring risk based on this analysis.
"""
                                try:
                                    overall_response = model.generate_content(overall_report_instruction_prompt)
                                    st.session_state.gemini_overall_report = overall_response.text.strip()
                                    st.markdown(st.session_state.gemini_overall_report)
                                    status_placeholder.success("Aggregate report generated successfully!")
                                except Exception as e:
                                    st.error(f"Error generating overall report with Gemini: {e}")
                                    status_placeholder.error("Failed to generate aggregate report.")
                            else:
                                st.warning("No individual row results to generate an aggregate report.")
                                status_placeholder.empty()
                else:
                    st.warning("Gemini model not initialized. Please check your API key.")
            else:
                st.info("Upload a CSV and select rows to run the Gemini Model.")
        except KeyError:
            st.info("Please add your Gemini API Key to your `secrets.toml` file to enable this model.")

    with col2:
        st.subheader("Pure Stats Model (Placeholder)")
        st.info("This column is reserved for the ML Model.")

    st.header("CSV Actions")
    col_actions1, col_actions2 = st.columns(2)
    with col_actions1:
        if st.button("Reset CSV to Original"):
            if st.session_state.df_original is not None:
                st.session_state.df_modified = st.session_state.df_original.copy()
                st.session_state.current_row_start_index = 0
                clear_gemini_results()
                st.success("CSV reset to original state!")
                st.dataframe(st.session_state.df_modified.head())
            else:
                st.warning("No original CSV to reset to.")
    with col_actions2:
        if st.session_state.df_modified is not None:
            csv_data = convert_df_to_csv(st.session_state.df_modified)
            st.download_button(
                label="Download Annotated CSV",
                data=csv_data,
                file_name="annotated_data.csv",
                mime="text/csv",
                help="Download the current version of the CSV with applied annotations."
            )
        else:
            st.info("Upload a CSV to enable download.")

else:
    st.info("Please upload a CSV file to begin.")
