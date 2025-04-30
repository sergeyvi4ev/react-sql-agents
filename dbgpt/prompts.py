ASSIGNMENT_PROMPT = """
You are investigating a crime case and your goal is to find the perpetrator.
Here is the assignment description:
{ASSIGNMENT_DESCRIPTION}
"""

PROMPT_DETECTIVE = '''Act as a detective. Your goal is to find a murderer. You have access to a police database. The data analyst is helping you to retriever the relevant information. These are facts so far with a hint where to start: 
    {facts}
    Inspect carefully the information in facts.
    You also know that the database has the following information in it:
    {database_info}
    Based on the found recent information,facts and available tables in the database generate an instruction for the data analyst what next clue can be searched in the database which will help the investigation. 
    Think step-by-step.
    If the last answer is error or no information, try another instruction.
    Start by retrieving the corresponding crime scene report from the police department’s database.
    Your instruction should be single and very specific so the analyst can generate SQL query from it. 
    Mention which table to use. Give only one instruction at a time.
    Always tell the analyst to not use queries retrieving all records from tables like "SELECT * FROM table_name"
    Always tell analyst to retrieve up to maximum 50 records by using LIMIT clause.
    Do not suggest any queries to the analyst, let him figure out the queries itself.
    '''

PROMPT_SENIOR_DETECTIVE = '''
Act as a detective. Your goal is to find a murderer. 
Based on given report of questions and answers provide who is the guilty one with argumentation. 
Use deductive thinking in think step-by-step. 
If you need to the investigation to go on output "CONTINUE" if opposite output "STOP INVESTIGATION"
The report: 
    {report}
    '''