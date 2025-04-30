import json
import time
import pandas as pd
from dbgpt.langchain.ResonseHandler import ReActAgentResponse
from langchain.prompts import ChatPromptTemplate
from dbgpt.langchain.create_agent import initialize_sql_agent
from langchain.chat_models import ChatOpenAI

def create_assignment(prompt_template, facts, db_info, llm):
    """Creates an assignment based on given facts and database information."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke({"facts": facts, "database_info": db_info})
    return response.content

def make_conclusion(prompt_template, report, llm):
    """Creates a final report based on collected facts."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke({"report": report})
    return response.content

def run(db, prompt_supervisor_template, prompt_manager_template, initial_task, model_name, temp, number_of_iterations):
    """
    Executes a series of iterations where an agent processes a question and generates an answer.

    Args:
        llm: The language model to use.
        db: The database object.
        prompt_supervisor_template: The template of supervisor.
        db_info: Information about the database.
        initial_task: The initial task or problem description.
        model_name: The name of the model to use.
        temp: The temperature setting for the model.
        number_of_iterations: The number of iterations to run.

    Returns:
        A tuple containing the final answer and the collected facts.
    """
    # enter the initial question into the report
    collected_facts = {'THE CRIME': {'question': "", 'answer': initial_task}}
    # initialize sql agent
    agent = initialize_sql_agent(model_name, temp, db)
    # initialized llm object with settings
    llm = ChatOpenAI(temperature=temp, model=model_name)
    # start agents procedure
    for i in range(number_of_iterations):
        try:
            print(f'--------------ITERATION----------------- {i}')
            print(f'KNOWN FACTS: {collected_facts}')

            # supervisor agent creates assignment
            question = create_assignment(
                prompt_template=prompt_supervisor_template,
                facts=collected_facts,
                db_info=db.table_info,
                llm=llm
            )

            print(f'\033[36mPASSING QUESTION TO THE ANALYST: {question}\033[0m')
            # run sql agent passing the question from supervisor-agent
            agent_response = agent.invoke({"input": question})
            sql_agent_answer = ReActAgentResponse(**agent_response).output
            error = None

        # handle errors
        except Exception as e:
            print(f"An error occurred: {e}")
            sql_agent_answer = None
            error = str(e)
            print(error)

        # record results
        iteration_info = {
            'question': question,
            'answer': sql_agent_answer,
            'error': error
        }

        # Store iteration info in collected facts
        collected_facts[f'Iteration {i}'] = iteration_info

        # Generate a conclusion based on the collected facts (agent-manager)
        manager_answer = make_conclusion(
            prompt_template=prompt_manager_template,
            report=collected_facts,
            llm=llm)

        print(f'FINAL_ANSWER: {manager_answer}')

        # Stop the loop if condition is met
        if "STOP INVESTIGATION" in manager_answer:
            print("Stopping iterations: Conclusion reached.")
            break

    # last iteration answer is the final answer
    final_answer = manager_answer

    return final_answer, collected_facts

def run_experiment(db, exp_params, report_path):
    results = []
    collected_facts = None
    for exp in exp_params:
        print("#############################")
        print(f"Run #:{exp['run']}")
        print(f"Parameters: Model: {exp['model']} Temperature: {exp['temp']} Iterations: {exp['iter_num']}")
        try:
            start_time = time.time()

            # Run the experiment
            final_answer, collected_facts = run(
                model_name=exp['model'],
                temp=exp['temp'],
                db=db,
                prompt_supervisor_template=exp['prompt_supervisor_template'],
                prompt_manager_template=exp['prompt_manager_template'],
                number_of_iterations=exp['iter_num'],
                initial_task=exp['input']
            )

            # Calculate the duration in minutes
            duration_minutes = (time.time() - start_time) / 60

            # Output to a JSON file
            with open(f'collected_facts_{exp["run"]}.json', 'w') as file:
                json.dump(collected_facts, file, indent=4)

            error_message = None  # No error occurred

            # Store results
            results.append({
                "run": exp.get("run"),
                "temp": exp['temp'],
                "model": exp['model'],
                "input": exp['input'],
                "final_answer": final_answer,
                "error_message": error_message,
                "memory": collected_facts,
                "duration_min": duration_minutes
            })

        except Exception as e:
            print(str(e))
            final_answer = None
            collected_facts = {}
            error_message = str(e)

    # Print or log the final answer and collected facts
    print("Final Answer:", final_answer)
    print("Collected Facts:", json.dumps(collected_facts, indent=4))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(report_path, index=False)

