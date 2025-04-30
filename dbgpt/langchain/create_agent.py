from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI

def initialize_sql_agent(model_name, temp, db):
    """Initializes the SQL agent with given model and temperature."""
    llm = ChatOpenAI(temperature=temp, model=model_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type='zero-shot-react-description',
        verbose=True,
        handle_parsing_errors=True
    )
    return agent