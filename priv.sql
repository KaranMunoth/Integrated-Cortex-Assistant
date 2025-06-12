-- Switch to a role that has the authority to grant privileges
USE ROLE ACCOUNTADMIN;

-- Set the context to the correct database and schema
USE DATABASE cortex_search_tutorial_db;
USE SCHEMA public;

-- Grant SELECT privileges to allow data access
GRANT SELECT ON TABLE raw_text TO ROLE cortex_user_role;
GRANT SELECT ON TABLE doc_chunks TO ROLE cortex_user_role;

-- Grant USAGE privilege on the Cortex Search service
GRANT USAGE ON CORTEX SEARCH SERVICE fomc_meeting TO ROLE cortex_user_role;

-- (Optional) Verify access
SHOW CORTEX SEARCH SERVICES;
SHOW CORTEX SEARCH SERVICES IN DATABASE cortex_search_tutorial_db;
SHOW CORTEX SEARCH SERVICES IN SCHEMA cortex_search_tutorial_db.public;
SHOW GRANTS TO ROLE cortex_user_role;
SHOW CORTEX SEARCH SERVICES LIKE 'fomc_meeting';
📌 These grants are essential for allowing your integrated assistant to query Cortex Search and access the underlying data. Without these, the application will not return results properly.
