import pandas as pd
from dbgpt.langchain.ResonseHandler import ReActAgentResponse
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI



def initialize_sql_agent(model_name, temp, db):
    # define llm
    llm = ChatOpenAI(temperature=temp, model=model_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # create agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type='zero-shot-react-description',
        verbose=True
    )
    return agent

def run_experiment(db, exp_params, report_path):

    results = []

    for exp in exp_params:
        print("#############################")
        print(f"Run #:{exp['run']}")
        print(f"Parameters: Model: {exp['model']} Temperature: {exp['temp']} Iterations: {exp['iter_num']}")
        agent = initialize_sql_agent(model_name=exp['model'], temp=exp['temp'], db=db)
        memory = ""
        print("\nFull Chain Memory:")
        print(memory)
        previous_input = exp["input"]  # Start with your original prompt

        for k in range(1, exp['iter_num']):
            try:
                # Build the prompt from previous input + known facts (memory)
                input_text = f"{previous_input} investigation progress so far: {memory} Continue Investigating"
                print(f"Chain #{k} Prompt:")
                print(input_text)

                # Invoke the agent with the constructed prompt
                agent_response = agent.invoke({"input": input_text})
                final_ans = ReActAgentResponse(**agent_response).output

                # Append the input and output to the overall memory
                memory += f"\nStep {k} Input: {previous_input}\nStep {k} Answer: {final_ans}\n"

                # The next iteration’s input becomes this iteration’s output
                previous_input = final_ans

            except Exception as e:
                error_message = f"Error in iteration {k}: {e}"
                print(error_message)
                # Optionally, track the error in the memory
                memory += f"\nStep {k} Error: {error_message}\n"
                # Continue to the next iteration

        # Store results
        results.append({
            "run": exp["run"],
            "temp": exp["temp"],
            "model": exp["model"],
            "input": exp["input"],
            "final_answer": final_ans,
            "error_message": error_message if 'error_message' in locals() else None,
            "memory": memory
        })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(report_path, index=False)