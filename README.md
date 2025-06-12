# 🤖 Integrated Assistant

An AI-powered assistant that integrates multiple domain-specific agents (like Collections, Disputes, AR, and FOMC) to provide intelligent insights and responses through a conversational interface. Built on top of Snowflake Cortex, Streamlit, and custom agent frameworks.

> ⚠️ **Important Prerequisites**  
Before using this repository, please **set up and configure** the following repositories:

1. [`FOMC_CORTEX_CHATBOT`](https://github.com/KaranMunoth/FOMC_CORTEX_CHATBOT)  
   Enables FOMC-related query handling via semantic search and contextual AI.

2. [`Cortex_Analyst_Chatbot`](https://github.com/KaranMunoth/Cortex_Analyst_Chatbot)  
   Powers the backend with Snowflake Cortex Analyst APIs for semantic querying and chart generation.

These projects are dependencies and must be initialized before running the integrated assistant.

---

## 🚀 Features

- 💬 **Chat-style interface** for natural language queries
- 🧠 **Cortex-powered agents** that understand context and generate SQL or summaries
- 📊 **Dynamic charting and data visualization**
- 🧾 **Multi-agent architecture** (Collections, Disputes, FOMC, etc.)
- 🛡️ **Secure integration** with Snowflake data warehouse
- 🔍 **Context-aware responses** using Snowflake Cortex Search and Analyst
