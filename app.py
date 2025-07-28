import streamlit as st
import pandas as pd
import io
import os
import google.generativeai as genai
import re
import json
import plotly.express as px
from collections import Counter


try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.warning("GEMINI_API_KEY not found in Streamlit Secrets. "
               "Please add it to .streamlit/secrets.toml or set it as an environment variable.")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

GEMINI_MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(layout="wide", page_title="Attrition & Hiring Risk Analyzer")

st.title("Attrition & Hiring Risk Analyzer")
st.markdown("### Compare AI Model Outputs Using Your CSV Data")
st.write("Upload your employee data, apply hypothetical changes, and see how different models classify hiring and attrition risks.")

DECISION_TREE_RULEBOOK = """
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

SCORE_COLUMNS_FOR_PLOT = [
    'Score', 'Conscientious', 'Organized', 'Integrity',
    'Withholding', 'Manipulative', 'Work Ethic/Duty', 'Achievement', 'Achor Cherry Picking'
]

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
    # Only show warning if API key is the default placeholder or not found in secrets
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.warning("Please replace 'YOUR_GEMINI_API_KEY_HERE' in your code or "
                   "set GEMINI_API_KEY in .streamlit/secrets.toml to enable model functionality.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
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


with st.container(border=True):
    st.header("Step 1: Upload Your CSV Data")
    st.write("Upload your employee data in CSV format. Ensure it contains relevant columns such as 'Score', 'GYR', 'Integrity', 'Withholding', 'Manipulative', 'Work Ethic/Duty', and 'Achievement' for accurate risk classification.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None and st.session_state.df_original is None:
        process_csv_upload(uploaded_file)
    elif uploaded_file is not None and uploaded_file.name != getattr(st.session_state.get('last_uploaded_filename'), 'name', ''):
        st.session_state.last_uploaded_filename = uploaded_file
        process_csv_upload(uploaded_file)
    

if st.session_state.df_modified is not None:
    with st.container(border=True):
        st.header("Step 2: Prepare Your Data for Analysis")
        st.write("Before running the models, you can select which rows to analyze and even modify numerical values to see how different scenarios impact the risk classifications.")
        st.subheader("Select Rows for Model Processing")
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

        st.subheader("Annotate Numerical Values in CSV")
        st.write("Adjust numerical values in selected columns by a percentage to observe potential changes in risk classifications.")
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

    with st.container(border=True):
        st.header("Step 3: Run Model Analysis & Review Results") 
        st.write("Execute the chosen models to classify your employee data and then review their respective outputs side-by-side for comparison.")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gemini Decision Tree Model")
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": 
                st.info("Please set your Gemini API key in `.streamlit/secrets.toml` to enable this model.")
            elif not current_rows_df.empty:
                model = get_gemini_model()
                if model:
                    if st.button("Run Gemini Model on Selected Rows", key="run_gemini_model"):
                        with st.spinner("Preparing to analyze..."):
                            per_row_summary_for_report = []
                            st.session_state.gemini_per_row_results = []

                            total_rows_to_process = len(current_rows_df)
                            status_placeholder = st.empty()

                            for idx, (_, row) in enumerate(current_rows_df.iterrows()):
                                current_processing_index = idx + 1
                                status_placeholder.info(f"Analyzing row {current_processing_index} of {total_rows_to_process}...")

                                row_csv_string = pd.DataFrame([row]).to_csv(index=False, header=True)
                                original_scores = {}
                                for col in SCORE_COLUMNS_FOR_PLOT:
                                    if col in row and pd.api.types.is_numeric_dtype(row[col]):
                                        original_scores[col] = row[col]
                                st.write(f"DEBUG for Row {row.name}:")
                                for debug_col in SCORE_COLUMNS_FOR_PLOT:
                                    is_present = debug_col in row
                                    val = None
                                    col_dtype = None
                                    is_numeric_check_passed = False
                                    
                                    if is_present:
                                        val = row[debug_col]
                                        col_dtype = type(val)
                                        is_numeric_check_passed = pd.api.types.is_numeric_dtype(val)
                                    
                                    st.write(f"  Column '{debug_col}': Present={is_present}, Value='{val}', Type={col_dtype}, IsNumericDtype={is_numeric_check_passed}")
                                    if is_present and not is_numeric_check_passed:
                                        st.write(f"    (Reason for non-numeric: Check CSV for non-number characters in this column!)")

                                st.write(f"  Final original_scores for Row {row.name}: {original_scores}")
                                st.write(f"  All scores present check result before plot: {all(col in original_scores for col in SCORE_COLUMNS_FOR_PLOT)}")
                                      
                                per_row_instruction_prompt = f"""
                                Your entire response for this row MUST be a single JSON object. Do NOT include any additional text, markdown formatting (like ```json), or conversation outside of the JSON object itself.

                                Using the "Definitive Attrition & Hiring Risk Model" rulebook provided below,
                                analyze this single row of CSV data.
                                Determine the exact "category" (e.g., "Category 7: The Volatile Performer")
                                that the employee falls into, applying the rules hierarchically from Category 7 down to 0.
                                
                                Then, provide a concise, one-line explanation *why* that specific category was assigned,
                                referencing the data criteria that led to that classification.
                                
                                Additionally, extract the `Risk Level` and `Estimated Tenure if Terminated` directly from the assigned category's description in the rulebook.

                                --- Definitive Attrition & Hiring Risk Model (Rulebook) ---
                                {DECISION_TREE_RULEBOOK}
                                --- End of Rulebook ---

                                --- Employee Data (CSV) ---
                                {row_csv_string}
                                --- End of Employee Data ---

                                The JSON object must have the following keys:
                                `row_id`: The original row identifier (e.g., "Row {row.name}")
                                `category_name`: The full category name (e.g., "Category 7: The Volatile Performer")
                                `risk_level`: The risk level from the rulebook for that category (e.g., "Highest Risk")
                                `estimated_tenure`: The estimated tenure from the rulebook for that category (e.g., "1 - 5 Months")
                                `explanation`: A concise, one-line explanation why this category was assigned.

                                Example JSON Output for Row 1:
                                {{
                                  "row_id": "Row 1",
                                  "category_name": "Category 4: The Burnout Risk",
                                  "risk_level": "Elevated",
                                  "estimated_tenure": "1 - 4 Months",
                                  "explanation": "Met criteria for high Conscientious (>80) and Achievement (>90) but very low Work Ethic/Duty (<15)."
                                }}
                                """
                                
                                try:
                                    response = model.generate_content(per_row_instruction_prompt)
                                    raw_response_text = response.text.strip() # Get the raw text

                                    # Attempt to clean and parse JSON
                                    json_analysis = {}
                                    try:
                                        # Try loading directly first (ideal)
                                        json_analysis = json.loads(raw_response_text)
                                    except json.JSONDecodeError:
                                        # If direct load fails, try to extract from markdown block
                                        json_match = re.search(r"```json\s*(\{.*\})\s*```", raw_response_text, re.DOTALL)
                                        if json_match:
                                            json_string = json_match.group(1)
                                            json_analysis = json.loads(json_string)
                                        else:
                                            # If still no valid JSON, raise error to be caught below
                                            raise json.JSONDecodeError("No valid JSON found in response.", raw_response_text, 0)


                                    st.session_state.gemini_per_row_results.append({
                                        "gemini_output": json_analysis,
                                        "original_row_data": original_scores # Store the extracted scores
                                    })
                                    per_row_summary_for_report.append(json.dumps(json_analysis))

                                except json.JSONDecodeError as e:
                                    st.error(f"Error decoding JSON for row {row.name}: {e}\nRaw response: {raw_response_text}")
                                    st.session_state.gemini_per_row_results.append({"gemini_output": {"row_id": f"Row {row.name}", "error": f"JSON decode error: {e}"}, "original_row_data": original_scores})
                                    per_row_summary_for_report.append(f"Error for Row {row.name}: JSON decode error")
                                except Exception as e:
                                    st.error(f"Error processing row {row.name} with Gemini: {e}")
                                    st.session_state.gemini_per_row_results.append({"gemini_output": {"row_id": f"Row {row.name}", "error": f"General error: {e}"}, "original_row_data": original_scores})
                                    per_row_summary_for_report.append(f"Error for Row {row.name}: General error")

                            status_placeholder.empty()

                            if per_row_summary_for_report:
                                status_placeholder.info("Generating comprehensive aggregate report...")
                                overall_report_instruction_prompt = f"""
                                You have just classified a set of employee data rows into risk categories
                                using the "Definitive Attrition & Hiring Risk Model" rulebook.

                                Here is the immutable rulebook you used:
                                --- Definitive Attrition & Hiring Risk Model (Rulebook) ---
                                {DECISION_TREE_RULEBOOK}
                                --- End of Rulebook ---

                                Here are the individual classification results for each employee/row, provided as a list of JSON objects (each object represents one row's analysis):
                                --- Per-Row Classification Results (JSON list) ---
                                {per_row_summary_for_report}
                                --- End of Per-Row Classification Results ---

                                Based on the rulebook and these individual classification results,
                                generate a comprehensive analysis report. Your report should be well-structured, clear,
                                and easy to understand for a hiring manager or HR professional.
                                Pay close attention to the 'category_name', 'risk_level', and 'estimated_tenure' fields in the JSON objects for your analysis.

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
                                Discuss how hypothetical changes to the input data for *any* employee would causally impact their classification across risk categories. For each mentioned metric (e.g., 'Score', 'Manipulative', 'GYR', 'Work Ethic/Duty', 'Integrity', 'Conscientious', 'Organized', 'Achievement'):
                                - Explain how increasing/decreasing its value could shift an employee from one category to another.
                                - Provide concrete, concise examples that directly reference the rulebook's criteria for category transitions.

                                **6. Conclusion:**
                                Provide a concise concluding summary of the report's main findings, emphasizing the most critical takeaways and strategic implications for managing attrition and hiring risk based on this analysis.
                                """
                                try:
                                    overall_response = model.generate_content(overall_report_instruction_prompt)
                                    st.session_state.gemini_overall_report = overall_response.text.strip()
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

            st.markdown("---")
            st.subheader("Gemini Model Analysis Results:")
            st.write("Explore the detailed classifications for each row and an aggregate report summarizing the findings.")

            if st.session_state.gemini_per_row_results:
                st.subheader("Per-Row Analysis Details:")

                with st.container(height=300, border=True):
                    if st.session_state.gemini_per_row_results:
                        for idx, item_data in enumerate(st.session_state.gemini_per_row_results):
                            display_row_index = st.session_state.current_row_start_index + idx
                            
                            # Access the nested data
                            gemini_output = item_data.get('gemini_output', {})
                            original_row_data = item_data.get('original_row_data', {})

                            # Extract details for display from gemini_output
                            category_name = gemini_output.get('category_name', 'N/A Category')
                            explanation = gemini_output.get('explanation', 'No explanation provided.')
                            
                            expander_title_suffix = category_name
                            display_content = f"**Risk Level:** {gemini_output.get('risk_level', 'N/A')}\n\n" \
                                              f"**Estimated Tenure if Terminated:** {gemini_output.get('estimated_tenure', 'N/A')}\n\n" \
                                              f"**Explanation:** {explanation}"

                            with st.expander(f"Row {display_row_index}: {expander_title_suffix}"):
                                st.markdown(display_content)
                                
                                # --- START ADDITION: Individual Score Plot ---
                                # Check if all required score columns are present and if there's any data
                                all_scores_present = all(col in original_row_data for col in SCORE_COLUMNS_FOR_PLOT)
                                
                                if all_scores_present and original_row_data:
                                    st.markdown("---") # Separator for visual clarity
                                    st.markdown("**Individual Score Profile:**")
                                    
                                    # Prepare data for this specific row's bar chart
                                    # Filter to ensure only defined SCORE_COLUMNS_FOR_PLOT are used
                                    plot_data_for_row = []
                                    for attr in SCORE_COLUMNS_FOR_PLOT:
                                        if attr in original_row_data: # Double check presence
                                            plot_data_for_row.append({'Attribute': attr, 'Value': original_row_data[attr]})
                                    
                                    if plot_data_for_row: # Ensure there's data to plot
                                        plot_df = pd.DataFrame(plot_data_for_row)
                                        
                                        fig_scores = px.bar(
                                            plot_df,
                                            x='Value',
                                            y='Attribute',
                                            orientation='h',
                                            labels={'Value': 'Score Value', 'Attribute': 'Attribute'},
                                            height=300, # Small plot
                                            range_x=[0, 100], # Standardize score range from 0 to 100
                                            color='Value', # Color bars based on their value
                                            color_continuous_scale=px.colors.sequential.Plasma # Use a continuous color scale
                                        )
                                        fig_scores.update_layout(
                                            showlegend=False, # No legend needed for single-row plot
                                            margin=dict(l=0, r=0, t=0, b=0), # Tight margins
                                            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                                            paper_bgcolor='rgba(0,0,0,0)' # Transparent paper background
                                        )
                                        fig_scores.update_xaxes(showgrid=False) # Hide x-axis grid lines for cleaner look
                                        fig_scores.update_yaxes(showgrid=False, categoryorder='total ascending') # Hide y-axis grid lines, sort attributes consistently
                                        
                                        st.plotly_chart(fig_scores, use_container_width=True)
                                    else:
                                        st.info("No numerical score data found for plotting in this employee's row.")
                                else:
                                    st.info("Score profile not available for this employee (missing required score attributes).")
                                # --- END ADDITION: Individual Score Plot ---

                    else:
                        st.info("No per-row results to display yet. Run the Gemini Model to see results here.")


            if st.session_state.gemini_overall_report:
                st.subheader("Comprehensive Analysis Report")
                st.markdown(st.session_state.gemini_overall_report)
            else:
                st.info("After running the Gemini model, a comprehensive analysis report along with some graphs will appear here, offering insights and recommendations.")
            
            # --- Start Graph Generation: Risk Category Distribution ---
            if st.session_state.gemini_per_row_results:
                st.markdown("---")
                st.subheader("Risk Category Distribution")
                st.write("This chart shows the number of employees falling into each risk category based on the Gemini Model's classification.")

                # Define the explicit order of categories by risk level for plotting
                category_order = [
                    "Category 7: The Volatile Performer",
                    "Category 6: The Mismatch",
                    "Category 5: The High-Friction Employee",
                    "Category 4: The Burnout Risk",
                    "Category 3: The Questionable Hire",
                    "Category 2: The Disengaged Professional",
                    "Category 1: The Apathetic Hire",
                    "Category 0: The Steady Performer"
                ]

                # Define a color map for risk levels
                risk_color_map = {
                    "Highest Risk": "darkred",
                    "Critical": "red",
                    "High": "orangered",
                    "Elevated": "orange",
                    "Moderate-High": "gold",
                    "Moderate": "yellowgreen",
                    "Moderate-Low": "limegreen",
                    "Low": "darkgreen",
                    "N/A": "gray" # Fallback for categories not explicitly mapped
                }

                # Prepare data for the bar chart
                category_counts = Counter()
                for result_dict in st.session_state.gemini_per_row_results:
                    category_name = result_dict.get('category_name')
                    if category_name:
                        category_counts[category_name] += 1
                    else:
                        category_counts["Error/Missing Category"] += 1

                # Convert Counter to DataFrame
                df_category_distribution = pd.DataFrame(category_counts.items(), columns=['Category', 'Count'])

                # Add Risk Level for coloring and sorting
                category_risk_mapping = {}
                for result_dict in st.session_state.gemini_per_row_results:
                    category = result_dict.get('category_name')
                    risk_level = result_dict.get('risk_level')
                    if category and risk_level:
                        category_risk_mapping[category] = risk_level
                
                df_category_distribution['Risk_Level'] = df_category_distribution['Category'].map(category_risk_mapping).fillna("N/A")

                # Ensure all categories from the order are in the DataFrame, even if count is 0
                full_categories = pd.DataFrame({'Category': category_order})
                df_category_distribution = pd.merge(full_categories, df_category_distribution, on='Category', how='left').fillna({'Count': 0, 'Risk_Level': 'N/A'})
                
                # Sort categories for consistent plotting order (highest risk at top for horizontal chart)
                df_category_distribution['Category'] = pd.Categorical(
                    df_category_distribution['Category'],
                    categories=category_order,
                    ordered=True
                )
                df_category_distribution = df_category_distribution.sort_values('Category', ascending=False) 

                # Create the bar chart
                fig_bar = px.bar(
                    df_category_distribution,
                    x='Count',
                    y='Category',
                    title='Number of Employees Per Risk Category',
                    color='Risk_Level', # Color by Risk Level
                    color_discrete_map=risk_color_map, # Apply custom color map
                    orientation='h', # Horizontal bars
                    text='Count', # Show count on bars
                    labels={'Category': 'Risk Category', 'Count': 'Number of Employees'}
                )

                fig_bar.update_traces(textposition='outside') # Position text outside bars
                fig_bar.update_layout(
                    showlegend=True,
                    # Ensure y-axis order matches our specified order
                    yaxis={'categoryorder':'array', 'categoryarray':list(df_category_distribution['Category'].astype(str))},
                    xaxis_title="Number of Employees",
                    margin=dict(l=0, r=0, t=40, b=0), # Adjust margins for better fit
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)' # Transparent paper background
                )
                fig_bar.update_xaxes(showgrid=False) # Hide x-axis grid lines
                fig_bar.update_yaxes(showgrid=False) # Hide y-axis grid lines

                st.plotly_chart(fig_bar, use_container_width=True)
                
        with col2:
            st.subheader("Pure Stats Model (Coming Soon)")
            st.info("This section will feature an advanced Machine Learning model for comparison, offering insights based on statistical patterns and predictive analytics.")
            st.markdown("---") 
            st.subheader("Pure Stats Model Analysis Results:") 
            st.write("Results from the Pure Stats Model will be displayed here for side-by-side comparison with the Gemini Model's output.")
            st.info("Run the Pure Stats Model (when available) to see analysis results here.")

    with st.container(border=True):
        st.header("CSV Actions")
        st.write("Manage your uploaded CSV data. You can revert to the original data or download the currently annotated version.") # Added description
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
