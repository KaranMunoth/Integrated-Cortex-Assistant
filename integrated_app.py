"""
Integrated Cortex Assistant
============================
This app combines Cortex Analyst (structured data queries) and FOMC Assistant (document search)
into a unified interface for comprehensive data analysis and document retrieval.
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import _snowflake
import pandas as pd
import streamlit as st
from snowflake.core import Root
from snowflake.cortex import Complete
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

# Configuration
MODELS = [
    "mistral-large",
    "llama3-70b",
    "llama3-8b",
]

# Cortex Analyst Configuration
AVAILABLE_SEMANTIC_MODELS_PATHS = [
    "CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.RAW_DATA/revenue_timeseries.yaml"
]
ANALYST_API_ENDPOINT = "/api/v2/cortex/analyst/message"
FEEDBACK_API_ENDPOINT = "/api/v2/cortex/analyst/feedback"
API_TIMEOUT = 50000

# FOMC Configuration
FOMC_DATABASE = "cortex_search_tutorial_db"
FOMC_SCHEMA = "public"
FOMC_SEARCH_SERVICE = "fomc_meeting"

# Initialize session
session = get_active_session()
root = Root(session)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Integrated Cortex Assistant",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()
    show_header_and_sidebar()

    # Main interface with dedicated tabs for each assistant
    tab_analyst, tab_fomc = st.tabs([
        "📊 Data Analysis (Cortex Analyst)",
        "📄 Document Search (FOMC)"
    ])

    with tab_analyst:
        cortex_analyst_interface()

    with tab_fomc:
        fomc_interface()


def initialize_session_state():
    """Initialize all session state variables."""
    if "analyst_messages" not in st.session_state:
        st.session_state.analyst_messages = []
    if "fomc_messages" not in st.session_state:
        st.session_state.fomc_messages = []
    if "active_suggestion" not in st.session_state:
        st.session_state.active_suggestion = None
    if "warnings" not in st.session_state:
        st.session_state.warnings = []
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = {}
    if "clear_conversation" not in st.session_state:
        st.session_state.clear_conversation = False
    if "selected_semantic_model_path" not in st.session_state:
        st.session_state.selected_semantic_model_path = AVAILABLE_SEMANTIC_MODELS_PATHS[0]
    if "selected_fomc_search_service" not in st.session_state:
        st.session_state.selected_fomc_search_service = FOMC_SEARCH_SERVICE


def show_header_and_sidebar():
    """Display header and sidebar configuration."""
    st.title("Integrated Cortex Assistant")
    st.markdown(
        "**AI Assistant** combining structured data analysis and document search capabilities."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection
        st.selectbox(
            "🧠 Select LLM Model:",
            MODELS,
            key="model_name",
            index=0
        )

        # Semantic Model Selection
        st.selectbox(
            "📊 Semantic Model:",
            AVAILABLE_SEMANTIC_MODELS_PATHS,
            format_func=lambda s: s.split("/")[-1],
            key="selected_semantic_model_path"
        )

        # Display the hardcoded FOMC Search Service
        st.markdown("---")
        st.markdown(f"**FOMC Search Service:**")
        st.markdown(f"`{FOMC_DATABASE}.{FOMC_SCHEMA}.{FOMC_SEARCH_SERVICE}`")
        st.markdown("---")


        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            st.number_input(
                "Context Chunks",
                value=5,
                key="num_retrieved_chunks",
                min_value=1,
                max_value=10,
                help="Number of document chunks to retrieve"
            )
            st.number_input(
                "Chat History Length",
                value=5,
                key="num_chat_messages",
                min_value=1,
                max_value=10,
                help="Number of previous messages to consider"
            )
            st.toggle("Debug Mode", key="debug", value=False)
            st.toggle("Use Chat History", key="use_chat_history", value=True)

        st.divider()

        # Clear buttons
        if st.button("🗑️ Clear All Conversations", use_container_width=True):
            st.session_state.analyst_messages = []
            st.session_state.fomc_messages = []
            st.session_state.warnings = []
            st.rerun()


def cortex_analyst_interface():
    """Cortex Analyst specific interface."""
    st.header("📊 Data Analysis with Cortex Analyst")

    # Display analyst conversation
    for idx, message in enumerate(st.session_state.analyst_messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            if role == "analyst":
                display_analyst_message(content, idx, message.get("request_id"))
            else:
                display_message_content(content, idx)

    # Analyst chat input
    if prompt := st.chat_input("Ask about your structured data..."):
        process_analyst_input(prompt)


def fomc_interface():
    """FOMC Assistant specific interface."""
    st.header("📄 FOMC Document Search")

    # Display FOMC conversation
    for message in st.session_state.fomc_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "references" in message:
                st.markdown(message["content"])
                st.markdown(message["references"])
            else:
                st.markdown(message["content"])

    # FOMC chat input
    if prompt := st.chat_input("Ask about FOMC documents..."):
        process_fomc_input(prompt)


def process_analyst_input(prompt: str):
    """Process Cortex Analyst input."""
    # Add user message
    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
    st.session_state.analyst_messages.append(new_user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get analyst response
    with st.chat_message("analyst"):
        with st.spinner("Generating analysis..."):
            response, error_msg = get_analyst_response(st.session_state.analyst_messages)
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response.get("request_id"),
                }

            st.session_state.analyst_messages.append(analyst_message)
            st.rerun()


def process_fomc_input(prompt: str):
    """Process FOMC input."""
    # Add user message
    st.session_state.fomc_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get FOMC response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = process_fomc_query(prompt)
            st.session_state.fomc_messages.append({
                "role": "assistant",
                "content": response["content"],
                "references": response["references"]
            })
            st.markdown(response["content"])
            st.markdown(response["references"])


def process_fomc_query(query: str) -> Dict:
    """Process FOMC query and return response with references."""
    # Create prompt for FOMC query
    prompt, results = create_fomc_prompt(query)

    # Generate response
    generated_response = Complete(st.session_state.model_name, prompt).replace("$", "\\$")

    # Build references
    references = build_references_table(results)

    return {
        "content": generated_response,
        "references": references
    }


def create_fomc_prompt(user_question: str) -> Tuple[str, List]:
    """Create prompt for FOMC query."""
    if st.session_state.use_chat_history and st.session_state.fomc_messages:
        chat_history = get_fomc_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context, results = query_fomc_search_service(question_summary)
        else:
            prompt_context, results = query_fomc_search_service(user_question)
    else:
        prompt_context, results = query_fomc_search_service(user_question)
        chat_history = ""

    prompt = f"""
        [INST]
        You are a helpful AI assistant specializing in FOMC (Federal Open Market Committee) documents.
        When a user asks you a question, you will be given context from FOMC documents between <context> and </context> tags.
        Use that context with the user's chat history to provide a comprehensive answer.

        Guidelines:
        - Be precise and factual
        - Quote specific information when relevant
        - If the context doesn't contain relevant information, say so clearly
        - Don't make assumptions beyond what's in the context

        <chat_history>
        {chat_history}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
        """
    return prompt, results


def query_fomc_search_service(query: str) -> Tuple[str, List]:
    """Query FOMC search service."""
    try:
        # Use the configured database and schema
        cortex_search_service = (
            root.databases[FOMC_DATABASE]
            .schemas[FOMC_SCHEMA]
            .cortex_search_services[st.session_state.selected_fomc_search_service]
        )

        context_documents = cortex_search_service.search(
            query,
            columns=["chunk", "file_url", "relative_path"],
            filter={"@and": [{"@eq": {"language": "English"}}]},
            limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results

        # Assume the text column is named 'chunk'
        search_col = "chunk"

        context_str = ""
        for i, r in enumerate(results):
            # Check if search_col exists in the result dictionary
            if search_col in r:
                context_str += f"Context document {i+1}: {r[search_col]} \n\n"
            else:
                 # Handle case where 'chunk' column is not found
                st.warning(f"Warning: Search column '{search_col}' not found in search result {i+1}.")


        if st.session_state.debug:
            st.sidebar.text_area("Retrieved Context", context_str, height=300)

        return context_str, results

    except Exception as e:
        st.error(f"Error querying search service: {str(e)}")
        return f"Error retrieving context: {str(e)}", []


def get_fomc_chat_history() -> str:
    """Get FOMC chat history as string."""
    if not st.session_state.fomc_messages:
        return ""

    start_index = max(0, len(st.session_state.fomc_messages) - st.session_state.num_chat_messages)
    recent_messages = st.session_state.fomc_messages[start_index:-1]  # Exclude current message

    history = ""
    for msg in recent_messages:
        role = msg["role"].title()
        content = msg["content"]
        history += f"{role}: {content}\n"

    return history


def make_chat_history_summary(chat_history: str, question: str) -> str:
    """Generate chat history summary."""
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """

    summary = Complete(st.session_state.model_name, prompt)

    if st.session_state.debug:
        st.sidebar.text_area("Chat History Summary", summary, height=150)

    return summary


def build_references_table(results: List) -> str:
    """Build markdown references table."""
    if not results:
        return ""

    markdown_table = "\n\n###### References \n\n| PDF Title | URL |\n|-------|-----|\n"
    for ref in results:
        title = ref.get('relative_path', 'Unknown')
        url = ref.get('file_url', '#')
        markdown_table += f"| {title} | [Link]({url}) |\n"

    return markdown_table


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """Get response from Cortex Analyst API."""
    request_body = {
        "messages": messages,
        "semantic_model_file": f"@{st.session_state.selected_semantic_model_path}",
    }

    try:
        resp = _snowflake.send_snow_api_request(
            "POST",
            ANALYST_API_ENDPOINT,
            {},
            {},
            request_body,
            None,
            API_TIMEOUT,
        )

        parsed_content = json.loads(resp["content"])

        if resp["status"] < 400:
            return parsed_content, None
        else:
            error_msg = f"API Error {resp['status']}: {parsed_content.get('message', 'Unknown error')}"
            return parsed_content, error_msg

    except Exception as e:
        return {}, f"Error calling Analyst API: {str(e)}"


def display_analyst_message(content: List[Dict], message_index: int, request_id: str = None):
    """Display Cortex Analyst message."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(suggestion, key=f"suggestion_{message_index}_{suggestion_index}"):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            display_sql_results(item["statement"], message_index, item.get("confidence"), request_id)


def display_message_content(content: List[Dict], message_index: int):
    """Display general message content."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])


def display_sql_results(sql: str, message_index: int, confidence: dict = None, request_id: str = None):
    """Display SQL query and results."""
    with st.expander("SQL Query", expanded=False):
        st.code(sql, language="sql")
        if confidence:
            display_sql_confidence(confidence)

    with st.expander("Results", expanded=True):
        with st.spinner("Executing SQL..."):
            df, error = get_query_exec_result(sql)
            if df is None:
                st.error(f"Query execution failed: {error}")
            elif df.empty:
                st.info("Query returned no data")
            else:
                data_tab, chart_tab = st.tabs(["Data 📄", "Chart 📉"])
                with data_tab:
                    st.dataframe(df, use_container_width=True)
                with chart_tab:
                    display_chart_options(df, message_index)

    if request_id:
        display_feedback_section(request_id)


def display_sql_confidence(confidence: dict):
    """Display SQL confidence information."""
    if not confidence:
        return

    verified_query = confidence.get("verified_query_used")
    with st.popover("Verified Query Info"):
        if verified_query is None:
            st.text("No verified query was used for this SQL generation")
        else:
            st.text(f"Name: {verified_query['name']}")
            st.text(f"Question: {verified_query['question']}")
            st.text(f"Verified by: {verified_query['verified_by']}")
            st.text(f"Verified at: {datetime.fromtimestamp(verified_query['verified_at'])}")
            st.code(verified_query["sql"], language="sql")


def display_chart_options(df: pd.DataFrame, message_index: int):
    """Display chart options."""
    if len(df.columns) >= 2:
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X axis", df.columns, key=f"x_col_{message_index}")
        y_cols = [col for col in df.columns if col != x_col]
        y_col = col2.selectbox("Y axis", y_cols, key=f"y_col_{message_index}")

        chart_type = st.selectbox(
            "Chart type",
            ["Line Chart 📈", "Bar Chart 📊"],
            key=f"chart_type_{message_index}"
        )

        try:
            if chart_type == "Line Chart 📈":
                st.line_chart(df.set_index(x_col)[y_col])
            else:
                st.bar_chart(df.set_index(x_col)[y_col])
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    else:
        st.info("Need at least 2 columns for charting")


def display_feedback_section(request_id: str):
    """Display feedback section."""
    with st.popover("📝 Feedback"):
        if request_id not in st.session_state.form_submitted:
            with st.form(f"feedback_{request_id}"):
                rating = st.radio("Rate this response:", ["👍 Good", "👎 Poor"], horizontal=True)
                feedback_text = st.text_area("Additional feedback (optional)")

                if st.form_submit_button("Submit Feedback"):
                    positive = rating == "👍 Good"
                    error = submit_feedback(request_id, positive, feedback_text)
                    st.session_state.form_submitted[request_id] = {"error": error}
                    st.rerun()
        else:
            result = st.session_state.form_submitted[request_id]
            if result["error"] is None:
                st.success("Feedback submitted! ✅")
            else:
                st.error(f"Feedback failed: {result['error']}")


def submit_feedback(request_id: str, positive: bool, feedback_message: str) -> Optional[str]:
    """Submit feedback to API."""
    try:
        request_body = {
            "request_id": request_id,
            "positive": positive,
            "feedback_message": feedback_message,
        }

        resp = _snowflake.send_snow_api_request(
            "POST",
            FEEDBACK_API_ENDPOINT,
            {},
            {},
            request_body,
            None,
            API_TIMEOUT,
        )

        if resp["status"] == 200:
            return None
        else:
            return f"API Error: {resp['status']}"

    except Exception as e:
        return f"Error submitting feedback: {str(e)}"


@st.cache_data(show_spinner=False)
def get_query_exec_result(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute SQL query and return results."""
    try:
        df = session.sql(query).to_pandas()
        return df, None
    except SnowparkSQLException as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    main()
