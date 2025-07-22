import streamlit as st
import pandas as pd
import io
import os
import google.generativeai as genai
import re

st.set_page_config(layout="wide", page_title="Attrition & Hiring Risk Analyzer")

st.title("Model Comparison UI")

DECISION_TREE_RULEBOOK = """
category 4: The Liability
Risk Level: Critical
Data Criteria: Score < 25 OR (GYR = RED AND Integrity < 40)
Profile Description: An individual who is either fundamentally incapable of
performing the job or possesses a critical character flaw that makes them an
immediate liability.
Actionable Insight: An unambiguous "Do Not Hire" signal. High probability
(>90%) of involuntary termination within 0-3 months.
category 3: The Deceiver
Risk Level: High
Data Criteria: GYR = RED AND (Withholding > 95 OR Manipulative > 90)
Profile Description: A skilled but highly toxic individual who uses
manipulation and information hoarding for personal gain. They are a severe
threat to team morale, trust, and productivity.
Actionable Insight: "Do Not Hire." The risk of poisoning team culture far
outweighs their skills. High probability (>80%) of involuntary termination within 2-6
months.
category 2: The High-Risk Hire
Risk Level: Elevated
Data Criteria: (GYR = RED AND Work Ethic/Duty < 40) OR (GYR =
GREEN/YELLOW AND Manipulative > 75 AND Score < 65) OR (GYR =
GREEN AND Work Ethic/Duty < 50)
Profile Description: Identifies three problematic types: the "Slacker"
(chronically poor effort), the "Unskilled Schemer" (covers incompetence with
politics), and the "Skilled, Apathetic Employee" (capable but lacks drive).
Actionable Insight: "Avoid Hiring." A significant drain on management time
with a high probability (~70%) of termination within 3-9 months.
Default Rule: Any GYR=RED employee not meeting the criteria for Level 4 or
3 is defaulted to this category.
category 1: The Gamble
Risk Level: Moderate
Data Criteria: GYR = YELLOW, OR (GYR = GREEN AND Score <
65), OR (GYR = GREEN AND Score >= 65 AND Manipulative > 75)
Profile Description: Identifies the "Average/Underperformer" and the "Skilled
but Political" hire. Their success is highly dependent on their direct manager
and work environment.
Actionable Insight: "Cautious Hire." A coin-toss with a ~50% long-term
success rate. Viable for simple, well-supervised roles only.
category 0b: The Superstar (Watchlist)
Risk Level: Low (Initial), but Moderate Flight Risk
Data Criteria: Meets all criteria for Level 0a AND (Score >= 88 AND
Achievement >= 90)
Profile Description: An exceptionally high-potential candidate who is also a
significant flight risk if not properly engaged, challenged, or recognized.
Actionable Insight: "Priority Hire, with a Proactive Retention Plan." The
hiring manager must be notified of the flight risk. Requires a 90-day
engagement plan.
category 0a: The Cornerstone (Standard)
Risk Level: Low
Data Criteria: GYR = GREEN AND Score >= 65 AND Manipulative <=
75 AND Work Ethic/Duty >= 50 (and does not meet L0b criteria).
Profile Description: The target hiring profile. Demonstrates capability, a
sound work ethic, and low behavioral risk.
Actionable Insight: "Confident Hire." This is your priority candidate pool for
stable, long-term success.
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

st.header("Gemini API Key Status")
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
