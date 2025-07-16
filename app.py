import streamlit as st
import pandas as pd
import google.generativeai as genai
import textwrap # For wrapping long text in the UI

# --- Configuration ---
# Configure the Gemini API key from Streamlit secrets
try:
    genai.configure(api_key=st.secrets["API_KEY"])
except AttributeError:
    st.error("Gemini API Key not found. Please set it in .streamlit/secrets.toml")
    st.stop() # Stop the app if key is not found

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.5-pro')

# --- Session State Initialization ---
# This ensures that variables persist across Streamlit reruns
if 'active_df' not in st.session_state:
    st.session_state.active_df = None
if 'terminated_df' not in st.session_state:
    st.session_state.terminated_df = None
if 'active_rows_sent_count' not in st.session_state:
    st.session_state.active_rows_sent_count = 0
if 'terminated_rows_sent_count' not in st.session_state:
    st.session_state.terminated_rows_sent_count = 0
if 'gemini_response_text' not in st.session_state:
    st.session_state.gemini_response_text = ""
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [] # To store messages sent to/from Gemini for context


# --- Helper Functions ---

def format_dataframe_for_gemini(df, start_index, num_rows, title):
    """
    Formats a slice of a DataFrame into a string for Gemini.
    """
    if df is None or num_rows <= 0:
        return f"No {title} data provided for this batch."

    end_index = min(start_index + num_rows, len(df))
    if start_index >= end_index:
        return f"No more {title} data available for this batch starting from row {start_index}."

    selected_rows = df.iloc[start_index:end_index]

    formatted_string = f"--- {title} Data (Rows {start_index} to {end_index-1}) ---\n"
    # Convert DataFrame to a string representation, e.g., CSV or Markdown table
    formatted_string += selected_rows.to_csv(index=False)
    formatted_string += "\n----------------------------------------\n"
    return formatted_string

def call_gemini_api(prompt_parts):
    """
    Sends the prompt to the Gemini API and handles the response.
    """
    st.spinner("Gemini is thinking...")
    try:
        # If there's existing conversation history, start a chat
        # Otherwise, just generate content
        if st.session_state.conversation_history:
            chat_session = model.start_chat(history=st.session_state.conversation_history)
            response = chat_session.send_message(prompt_parts)
        else:
            response = model.generate_content(prompt_parts)

        # Append current interaction to history
        st.session_state.conversation_history.append({'role': 'user', 'parts': prompt_parts})
        st.session_state.conversation_history.append({'role': 'model', 'parts': response.text})

        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="Gemini Data Processor")
st.title("Gemini Data Processor with Iterative Input")

st.markdown("""
Upload your 'Active' and 'Terminated' CSV files.
Specify the number of rows you want to send in each batch to Gemini,
and provide your prompt. You can then edit Gemini's response and
send the next batch of rows for further processing.
""")

# --- File Upload Section ---
st.header("1. Upload Data Files")
col1, col2 = st.columns(2)

with col1:
    uploaded_active_file = st.file_uploader("Upload Active Records CSV", type="csv", key="active_upload")
    if uploaded_active_file is not None:
        try:
            st.session_state.active_df = pd.read_csv(uploaded_active_file)
            st.info(f"Active Records: {len(st.session_state.active_df)} rows loaded.")
            # Reset counters if new file uploaded
            st.session_state.active_rows_sent_count = 0
            st.session_state.gemini_response_text = ""
            st.session_state.conversation_history = []
        except Exception as e:
            st.error(f"Error reading Active CSV: {e}")

with col2:
    uploaded_terminated_file = st.file_uploader("Upload Terminated Records CSV", type="csv", key="terminated_upload")
    if uploaded_terminated_file is not None:
        try:
            st.session_state.terminated_df = pd.read_csv(uploaded_terminated_file)
            st.info(f"Terminated Records: {len(st.session_state.terminated_df)} rows loaded.")
            # Reset counters if new file uploaded
            st.session_state.terminated_rows_sent_count = 0
            st.session_state.gemini_response_text = ""
            st.session_state.conversation_history = []
        except Exception as e:
            st.error(f"Error reading Terminated CSV: {e}")

st.markdown("---")

# --- Row Selection & Prompt Section ---
st.header("2. Configure Batch & Prompt Gemini")

if st.session_state.active_df is not None or st.session_state.terminated_df is not None:
    col3, col4 = st.columns(2)

    with col3:
        max_active_rows = len(st.session_state.active_df) if st.session_state.active_df is not None else 0
        current_active_start = st.session_state.active_rows_sent_count
        active_rows_left = max(0, max_active_rows - current_active_start)

        st.subheader(f"Active Records (Remaining: {active_rows_left} rows)")
        if active_rows_left > 0:
            rows_to_send_active = st.number_input(
                "Number of Active rows to send in this batch:",
                min_value=0,
                max_value=active_rows_left,
                value=min(10, active_rows_left), # Default to 10 or remaining
                key="active_rows_input"
            )
        else:
            rows_to_send_active = 0
            st.write("No more active rows available to send.")

    with col4:
        max_terminated_rows = len(st.session_state.terminated_df) if st.session_state.terminated_df is not None else 0
        current_terminated_start = st.session_state.terminated_rows_sent_count
        terminated_rows_left = max(0, max_terminated_rows - current_terminated_start)

        st.subheader(f"Terminated Records (Remaining: {terminated_rows_left} rows)")
        if terminated_rows_left > 0:
            rows_to_send_terminated = st.number_input(
                "Number of Terminated rows to send in this batch:",
                min_value=0,
                max_value=terminated_rows_left,
                value=min(10, terminated_rows_left), # Default to 10 or remaining
                key="terminated_rows_input"
            )
        else:
            rows_to_send_terminated = 0
            st.write("No more terminated rows available to send.")

    st.subheader("Your Instructions for Gemini")
    user_prompt = st.text_area(
        "Enter your prompt (e.g., 'Analyze this data for trends', 'Summarize key differences'):",
        key="user_prompt_input",
        height=150,
        value="Analyze the provided active and terminated customer data. Identify any patterns, key differences, or insights that could be useful for customer retention or win-back strategies. Focus on common attributes like tenure, last activity, or service usage if available. Provide a concise summary." if not st.session_state.conversation_history else ""
    )

    if st.button("Send to Gemini (Initial/Next Batch)"):
        if (rows_to_send_active == 0 and rows_to_send_terminated == 0) and not st.session_state.conversation_history:
            st.warning("Please select some rows from at least one file or start with an initial prompt if no data is needed for the first turn.")
        elif not user_prompt.strip() and not st.session_state.conversation_history:
            st.warning("Please provide a prompt for Gemini.")
        else:
            # Prepare data slices for the current batch
            active_data_str = format_dataframe_for_gemini(
                st.session_state.active_df,
                st.session_state.active_rows_sent_count,
                rows_to_send_active,
                "Active Customers"
            )
            terminated_data_str = format_dataframe_for_gemini(
                st.session_state.terminated_df,
                st.session_state.terminated_rows_sent_count,
                rows_to_send_terminated,
                "Terminated Customers"
            )

            # Construct the prompt for Gemini
            prompt_parts = []

            # If this is not the first interaction and there's previous edited output, include it
            if st.session_state.gemini_response_text.strip() and st.session_state.conversation_history:
                 prompt_parts.append("--- User-Edited Previous Output ---\n" + st.session_state.gemini_response_text.strip() + "\n----------------------------------\n")
                 prompt_parts.append("Considering the above context and your previous analysis, along with the *new data* provided below:")


            # Add the user's current instructions
            if user_prompt.strip():
                if st.session_state.conversation_history: # If it's a follow-up, prefix with user's new instruction
                    prompt_parts.append(f"\nUser's next instruction: {user_prompt.strip()}\n")
                else: # For initial call
                    prompt_parts.append(user_prompt.strip())

            # Add the new batch of data
            if active_data_str.strip() != "No Active data provided for this batch.":
                prompt_parts.append(active_data_str)
            if terminated_data_str.strip() != "No Terminated data provided for this batch.":
                prompt_parts.append(terminated_data_str)

            # Add a clear separator if both data parts are present
            if active_data_str.strip() != "No Active data provided for this batch." and \
               terminated_data_str.strip() != "No Terminated data provided for this batch.":
                prompt_parts.insert(-1, "\n--- Combined Data Set ---\n")


            # Combine all parts into a single string for the API call initially
            final_prompt_string = "\n".join(prompt_parts)

            with st.spinner("Calling Gemini API..."):
                gemini_output = call_gemini_api(final_prompt_string)

                if gemini_output:
                    st.session_state.gemini_response_text = gemini_output
                    st.success("Gemini responded!")

                    # Update sent counts only after successful API call
                    st.session_state.active_rows_sent_count += rows_to_send_active
                    st.session_state.terminated_rows_sent_count += rows_to_send_terminated
                else:
                    st.error("Failed to get a response from Gemini.")

else:
    st.info("Please upload files to enable row selection and prompt functionality.")

st.markdown("---")

# --- Gemini Response Section ---
st.header("3. Gemini's Response (Editable)")

if st.session_state.gemini_response_text:
    st.session_state.gemini_response_text = st.text_area(
        "Gemini's Output (Edit if needed for next interaction):",
        value=st.session_state.gemini_response_text,
        height=400,
        key="gemini_output_editor"
    )
    st.info("You can edit the response above. When you send the next batch of rows, this edited text will be included as part of the context for Gemini.")
else:
    st.info("Gemini's response will appear here after you send your first query.")

# Optional: Display conversation history for debugging/transparency
if st.session_state.conversation_history:
    st.subheader("Conversation History (For context)")
    with st.expander("View full conversation history"):
        for i, turn in enumerate(st.session_state.conversation_history):
            role = "You" if turn['role'] == 'user' else "Gemini"
            st.markdown(f"**{role}:**")
            # Textwrap for better display of long content
            st.markdown(f"```\n{textwrap.fill(turn['parts'], width=100)}\n```") # Use textwrap for long parts
            st.markdown("---")
