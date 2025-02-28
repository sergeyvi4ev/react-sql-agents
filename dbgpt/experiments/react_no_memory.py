import pandas as pd
from dbgpt.langchain.ResonseHandler import ReActAgentResponse
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

def initialize_sql_agent(model_name, temp, db):
    # define llm
    llm = ChatOpenAI(temperature=temp, model=model_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # create agent
    agent_executor = create_sql_agent(llm=llm,
                                  toolkit=toolkit,
                                  verbose=True,
                                  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                  handle_parsing_error=True
                                  )
    return agent_executor

def run_experiment(db, exp_params, report_path):
    """

    :param db: SQLDatabase
    :param exp_params:
    :param report_path:
    """

    results = []

    for exp in exp_params:
        print("#############################")
        print(f"Run #:{exp['run']}")
        print(f"Parameters: Model: {exp['model']} Temperature: {exp['temp']}")
        agent = initialize_sql_agent(model_name=exp['model'], temp=exp['temp'], db=db)
        agent_response = agent.invoke({"input": exp['input']})
        final_ans = ReActAgentResponse(**agent_response).output

        # Store results
        results.append({
            "run": exp["run"],
            "temp": exp["temp"],
            "model": exp["model"],
            "input": exp["input"],
            "final_answer": final_ans
        })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(report_path, index=False)