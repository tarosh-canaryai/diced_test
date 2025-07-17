import streamlit as st
import pandas as pd
import google.generativeai as genai
import textwrap
import uuid

# --- Configuration ---
# Use Streamlit secrets for API key
API_KEY = st.secrets["API_KEY"]

try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
    else:
        st.warning("Gemini API Key is empty. Set 'GEMINI_API_KEY' in Streamlit secrets for full functionality. API calls will fail.")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Check your API key.")

model = None
if API_KEY:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}. Check your API key and network.")
else:
    st.info("Gemini model not initialized due to missing API key.")

def initialize_session_state():
    if 'active_df' not in st.session_state:
        st.session_state.active_df = None
    if 'terminated_df' not in st.session_state:
        st.session_state.terminated_df = None
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'current_turn_index' not in st.session_state:
        st.session_state.current_turn_index = 0 

    if not st.session_state.history and st.session_state.current_turn_index == 0:
        st.session_state.current_gemini_response_display = ""
        st.session_state.current_user_prompt_display = '''Analyze the provided active and terminated employee data.
For active employees, create distinct 'character profiles' or segments. These profiles should describe common attributes such as:
Tenure (length of employment if there is a termination date available otherwise duration signifies how long they have been working upto that point)
personality scores 
Any other relevant characteristics present in the data.

For each active employee profile, predict their likelihood of voluntary termination and an expected timeline for departure. Categorize the timeline into the following ranges:
0-3 months
3-6 months
6-12 months
12+ months

For terminated employees, analyze their characteristics (tenure at termination, role, etc.) and their termination_date (if available in the data) to identify patterns that contributed to their departure. Use these patterns to inform the predictions for active employees. If a termination_date column is present, prioritize it for understanding churn timing.

Provide a concise summary of your findings, highlighting key characteristics of high-risk profiles and the most common churn timelines. Structure your output clearly, perhaps with a section for each active employee profile found. (dont give an individual summary for each person instead group into profiles that follow similar trends.'''
    elif 'current_gemini_response_display' not in st.session_state:
         st.session_state.current_gemini_response_display = ""
    elif 'current_user_prompt_display' not in st.session_state: 
         st.session_state.current_user_prompt_display = ""

    if 'active_rows_to_send_input_value' not in st.session_state:
        st.session_state.active_rows_to_send_input_value = 0
    if 'terminated_rows_to_send_input_value' not in st.session_state:
        st.session_state.terminated_rows_to_send_input_value = 0

    if 'active_file_name' not in st.session_state:
        st.session_state.active_file_name = None
    if 'terminated_file_name' not in st.session_state:
        st.session_state.terminated_file_name = None

    if 'confirm_clear_visible' not in st.session_state:
        st.session_state.confirm_clear_visible = False


initialize_session_state()


def format_dataframe_for_gemini(df, start_index, num_rows, title):
    if df is None or num_rows <= 0:
        return f"No {title} data provided for this batch."

    end_index = min(start_index + num_rows, len(df))
    if start_index >= end_index:
        return f"No more {title} data available for this batch starting from row {start_index}."

    selected_rows = df.iloc[start_index:end_index]
    formatted_string = f"--- {title} Data (Rows {start_index} to {end_index-1}) ---\n"
    formatted_string += selected_rows.to_csv(index=False)
    formatted_string += "\n----------------------------------------\n"
    return formatted_string

def get_full_gemini_context_from_history():
    context = []
    
    max_history_to_include = st.session_state.current_turn_index + 1
    if st.session_state.current_turn_index == len(st.session_state.history): 
        max_history_to_include = len(st.session_state.history)

    for i in range(max_history_to_include):
        turn = st.session_state.history[i]
        context.append({'role': 'user', 'parts': turn['user_prompt']})
        context.append({'role': 'model', 'parts': str(turn['user_edited_response'])})
    return context

def send_to_gemini_callback():
    current_key_suffix = st.session_state.current_turn_index 

    user_prompt_for_turn = st.session_state[f"user_prompt_input_widget_{current_key_suffix}"]
    active_rows_for_turn = st.session_state[f"active_rows_input_{current_key_suffix}"]
    terminated_rows_for_turn = st.session_state[f"terminated_rows_input_{current_key_suffix}"]

    if model is None:
        st.error("Gemini model not initialized. Cannot send request.")
        return

    if st.session_state.current_turn_index < len(st.session_state.history):
        st.session_state.history[st.session_state.current_turn_index]['user_edited_response'] = \
            st.session_state[f"gemini_output_editor_widget_{current_key_suffix}"]
    
    st.session_state.history = st.session_state.history[:st.session_state.current_turn_index + 1]

    initial_active_rows_sent_count = 0
    initial_terminated_rows_sent_count = 0
    if st.session_state.history: 
        initial_active_rows_sent_count = st.session_state.history[-1]['active_rows_at_turn_end']
        initial_terminated_rows_sent_count = st.session_state.history[-1]['terminated_rows_at_turn_end']

    active_data_str = format_dataframe_for_gemini(
        st.session_state.active_df,
        initial_active_rows_sent_count,
        active_rows_for_turn,
        "Active Customers"
    )
    terminated_data_str = format_dataframe_for_gemini(
        st.session_state.terminated_df,
        initial_terminated_rows_sent_count,
        terminated_rows_for_turn,
        "Terminated Customers"
    )

    current_turn_prompt_parts = []

    if st.session_state.history: 
        prev_edited_response = st.session_state.history[-1]['user_edited_response']
        if prev_edited_response.strip():
            current_turn_prompt_parts.append(
                "--- User-Edited Previous Output ---\n" +
                prev_edited_response.strip() +
                "\n----------------------------------\n"
            )
            current_turn_prompt_parts.append("Considering the above context and your previous analysis, along with the *new data* provided below:")

    if user_prompt_for_turn.strip():
        if current_turn_prompt_parts:
            current_turn_prompt_parts.append(f"\nUser's next instruction: {user_prompt_for_turn.strip()}\n")
        else:
            current_turn_prompt_parts.append(user_prompt_for_turn.strip())

    if active_data_str.strip() != "No Active data provided for this batch.":
        current_turn_prompt_parts.append(active_data_str)
    if terminated_data_str.strip() != "No Terminated data provided for this batch.":
        current_turn_prompt_parts.append(terminated_data_str)

    if (active_data_str.strip() != "No Active data provided for this batch." and
        terminated_data_str.strip() != "No Terminated data provided for this batch."):
        if len(current_turn_prompt_parts) >= 2 and "--- User-Edited Previous Output ---" in current_turn_prompt_parts[0]: 
             current_turn_prompt_parts.insert(-2, "\n--- Combined Data Set ---\n")
        elif len(current_turn_prompt_parts) >= 2 and not st.session_state.history: 
             current_turn_prompt_parts.insert(1, "\n--- Combined Data Set ---\n")


    final_prompt_string = "\n".join(current_turn_prompt_parts)

    gemini_output = None
    try:
        with st.spinner("Calling Gemini API..."):
            full_context_for_gemini = get_full_gemini_context_from_history()
            
            chat_session = model.start_chat(history=full_context_for_gemini)
            response = chat_session.send_message(final_prompt_string)
            gemini_output = response.text

    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        gemini_output = None

    if gemini_output:
        new_turn_entry = {
            'turn_number': len(st.session_state.history) + 1,
            'user_prompt': user_prompt_for_turn, 
            'active_data_sent_start_index': initial_active_rows_sent_count,
            'active_data_sent_num_rows': active_rows_for_turn,
            'terminated_data_sent_start_index': initial_terminated_rows_sent_count,
            'terminated_data_sent_num_rows': terminated_rows_for_turn,
            'gemini_response': gemini_output,
            'user_edited_response': gemini_output, 
            'active_rows_at_turn_end': initial_active_rows_sent_count + active_rows_for_turn,
            'terminated_rows_at_turn_end': initial_terminated_rows_sent_count + terminated_rows_for_turn,
        }
        st.session_state.history.append(new_turn_entry)

        st.session_state.current_turn_index = len(st.session_state.history) - 1

        st.session_state.current_gemini_response_display = gemini_output
        st.session_state.current_user_prompt_display = new_turn_entry['user_prompt'] 

        st.session_state.active_rows_to_send_input_value = 0
        st.session_state.terminated_rows_to_send_input_value = 0

    else:
        st.error("Failed to get a response from Gemini.")

def reset_application_logic():
    for key in list(st.session_state.keys()): 
        del st.session_state[key]
    initialize_session_state() 

def reset_application_confirm_callback():
    st.session_state.confirm_clear_visible = False 
    reset_application_logic()

def show_confirm_clear_callback():
    st.session_state.confirm_clear_visible = True

def select_turn_for_display(turn_index):
    if st.session_state.current_turn_index < len(st.session_state.history):
        current_key_suffix = st.session_state.current_turn_index
        st.session_state.history[st.session_state.current_turn_index]['user_edited_response'] = \
            st.session_state[f"gemini_output_editor_widget_{current_key_suffix}"]

    st.session_state.current_turn_index = turn_index

    selected_turn = st.session_state.history[turn_index]
    st.session_state.current_gemini_response_display = selected_turn['user_edited_response']
    st.session_state.current_user_prompt_display = selected_turn['user_prompt']

    st.session_state.active_rows_to_send_input_value = 0
    st.session_state.terminated_rows_to_send_input_value = 0
    

def create_new_turn_mode():
    if st.session_state.current_turn_index < len(st.session_state.history):
        current_key_suffix = st.session_state.current_turn_index
        st.session_state.history[st.session_state.current_turn_index]['user_edited_response'] = \
            st.session_state[f"gemini_output_editor_widget_{current_key_suffix}"]
    
    st.session_state.current_turn_index = len(st.session_state.history)
    
    st.session_state.current_gemini_response_display = ""
    if st.session_state.history:
        st.session_state.current_user_prompt_display = st.session_state.history[-1]['user_prompt']
    else:
        st.session_state.current_user_prompt_display = "Analyze the provided active and terminated customer data. Identify any patterns, key differences, or insights that could be useful for customer retention or win-back strategies. Focus on common attributes like tenure, last activity, or service usage if available. Provide a concise summary."

    st.session_state.active_rows_to_send_input_value = 0
    st.session_state.terminated_rows_to_send_input_value = 0

# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="Gemini Data Processor")
st.title("Gemini Data Processor with Iterative Input")

st.markdown("""
Upload your 'Active' and 'Terminated' CSV files.
This app allows you to interactively analyze large datasets with Gemini in batches.
Each interaction is saved in the sidebar, and you can review past turns or start new ones.
""")

with st.sidebar:
    st.header("Global Controls")
    if st.button("Clear Data & Reset Application ", key="clear_all_button", on_click=show_confirm_clear_callback):
        pass

    if st.session_state.confirm_clear_visible:
        st.warning("Are you sure? This will delete uploaded data, conversation history and you will need to reupload your files during the next iteration.")
        if st.button("Confirm Clear All", key="confirm_clear_all_action", on_click=reset_application_confirm_callback):
            pass

    st.markdown("---")
    
    is_new_turn_mode_active = (st.session_state.current_turn_index == len(st.session_state.history))
    if is_new_turn_mode_active:
        st.markdown("**➕ New Turn (Active)**")
    else:
        if st.button("➕ Start New Turn", key="start_new_turn_sidebar_button", on_click=create_new_turn_mode):
            pass
        
    st.markdown('<p style="font-size:12px;">(Resets rows used in the csv files and allows you to start a new turn with no context.)</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## Previous Analysis Turns")

    if not st.session_state.history:
        st.info("No turns completed yet.")
    else:
        for i in range(len(st.session_state.history) - 1, -1, -1):
            turn = st.session_state.history[i]
            is_selected_historical_turn = (i == st.session_state.current_turn_index) and not is_new_turn_mode_active
            button_label = f"**Turn {turn['turn_number']}**"
            
            if is_selected_historical_turn:
                st.markdown(f"➡️ {button_label} (Selected)")
            else:
                if st.button(button_label, key=f"select_turn_{turn['turn_number']}_{uuid.uuid4()}", on_click=select_turn_for_display, args=(i,)):
                    pass


st.markdown("---") 

st.header("1. Upload Data Files")
col1, col2 = st.columns(2)

with col1:
    uploaded_active_file = st.file_uploader("Upload Active Records CSV", type="csv", key="active_upload")
    if uploaded_active_file is not None:
        if st.session_state.active_df is None or uploaded_active_file.name != st.session_state.get('active_file_name'):
            try:
                st.session_state.active_df = pd.read_csv(uploaded_active_file)
                st.session_state.active_file_name = uploaded_active_file.name
                st.info(f"Active Records: {len(st.session_state.active_df)} rows loaded.")
            except Exception as e:
                st.error(f"Error reading Active CSV: {e}")
    elif uploaded_active_file is None and st.session_state.active_df is not None:
        st.session_state.active_df = None
        st.session_state.active_file_name = None


with col2:
    uploaded_terminated_file = st.file_uploader("Upload Terminated Records CSV", type="csv", key="terminated_upload")
    if uploaded_terminated_file is not None:
        if st.session_state.terminated_df is None or uploaded_terminated_file.name != st.session_state.get('terminated_file_name'):
            try:
                st.session_state.terminated_df = pd.read_csv(uploaded_terminated_file)
                st.session_state.terminated_file_name = uploaded_terminated_file.name
                st.info(f"Terminated Records: {len(st.session_state.terminated_df)} rows loaded.")
            except Exception as e:
                st.error(f"Error reading Terminated CSV: {e}")
    elif uploaded_terminated_file is None and st.session_state.terminated_df is not None:
        st.session_state.terminated_df = None
        st.session_state.terminated_file_name = None


st.markdown("---")

st.header("2. Configure Current Turn / Send to Gemini")

if st.session_state.current_turn_index < len(st.session_state.history):
    active_turn_data = st.session_state.history[st.session_state.current_turn_index]
    st.subheader(f"Viewing Turn {active_turn_data['turn_number']} Details")
    
    st.markdown("### **User Prompt for this Turn:**")
    st.markdown(active_turn_data['user_prompt'])

    st.markdown("### **Gemini's Raw Response for this Turn:**")
    st.markdown(active_turn_data['gemini_response'])

    st.markdown(f"**Active Rows Processed in this turn:** Rows {active_turn_data['active_data_sent_start_index']} to {active_turn_data['active_data_sent_start_index'] + active_turn_data['active_data_sent_num_rows'] - 1} ({active_turn_data['active_data_sent_num_rows']} rows)")
    st.markdown(f"**Terminated Rows Processed in this turn:** Rows {active_turn_data['terminated_data_sent_start_index']} to {active_turn_data['terminated_data_sent_start_index'] + active_turn_data['terminated_data_sent_num_rows'] - 1} ({active_turn_data['terminated_data_sent_num_rows']} rows)")

    st.markdown("---")
    st.subheader("Prepare for Next Interaction (or create new turn)")

else:
    st.subheader("Setting up New Turn")


last_turn_end_active_rows = 0
last_turn_end_terminated_rows = 0

if st.session_state.history and st.session_state.current_turn_index < len(st.session_state.history):
    selected_turn_data_source = st.session_state.history[st.session_state.current_turn_index]
    last_turn_end_active_rows = selected_turn_data_source['active_rows_at_turn_end']
    last_turn_end_terminated_rows = selected_turn_data_source['terminated_rows_at_turn_end']

max_active_rows = len(st.session_state.active_df) if st.session_state.active_df is not None else 0
max_terminated_rows = len(st.session_state.terminated_df) if st.session_state.terminated_df is not None else 0

active_rows_left = max(0, max_active_rows - last_turn_end_active_rows)
terminated_rows_left = max(0, max_terminated_rows - last_turn_end_terminated_rows)

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"**Active Records (Remaining: {active_rows_left} rows)**")
    st.number_input(
        "Number of Active rows to send in this batch:",
        min_value=0,
        max_value=active_rows_left,
        value=int(st.session_state.active_rows_to_send_input_value or min(10, active_rows_left)),
        key=f"active_rows_input_{st.session_state.current_turn_index}" 
    )

with col4:
    st.markdown(f"**Terminated Records (Remaining: {terminated_rows_left} rows)**")
    st.number_input(
        "Number of Terminated rows to send in this batch:",
        min_value=0,
        max_value=terminated_rows_left,
        value=int(st.session_state.terminated_rows_to_send_input_value or min(10, terminated_rows_left)),
        key=f"terminated_rows_input_{st.session_state.current_turn_index}" 
    )

st.markdown("---")

st.markdown("### Gemini's Response (Editable for Next Turn's Context)")
st.text_area(
    "Edit output or provide additional context:",
    value=st.session_state.current_gemini_response_display,
    height=300,
    key=f"gemini_output_editor_widget_{st.session_state.current_turn_index}" 
)
st.info("The content in this box will be included as 'User-Edited Previous Output' for the next API call to Gemini.")

st.markdown("---")

st.markdown("### Your Instructions for Gemini for the Next Step")
st.text_area(
    "Enter your prompt (e.g., 'Analyze this data for trends', 'Summarize key differences'):",
    key=f"user_prompt_input_widget_{st.session_state.current_turn_index}", 
    height=150,
    value=st.session_state.current_user_prompt_display 
)

if st.button("Send to Gemini (Process Next Batch)", on_click=send_to_gemini_callback):
    pass

else:
    st.info("Please upload files to enable data analysis functionality.")
