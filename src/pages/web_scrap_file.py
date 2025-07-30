import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pydantic import BaseModel, Field
from typing import Optional, List
from io import StringIO
import csv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent



class RobotSpec(BaseModel):
    robot_name: Optional[str] = Field(None, alias="Robot Name")
    company: Optional[str] = Field(None, alias="Company")
    country: Optional[str] = Field(None, alias="Country")
    region: Optional[str] = Field(None, alias="Region")
    year_unveiled: Optional[int] = Field(None, alias="Year Unveiled")
    primary_use_case: Optional[str] = Field(None, alias="Primary Use-Case")
    status: Optional[str] = Field(None, alias="Status")
    price_usd: Optional[float] = Field(None, alias="Price (USD)")
    production_capacity_units_per_year: Optional[int] = Field(None, alias="Production Capacity (units/year)")
    height_cm: Optional[float] = Field(None, alias="Height (in cm)")
    weight_kg: Optional[float] = Field(None, alias="Weight (in kg)")
    mobility_type: Optional[str] = Field(None, alias="Mobility Type")
    speed_m_per_s: Optional[float] = Field(None, alias="Speed (in m/s)")
    autonomy_hours: Optional[float] = Field(None, alias="Autonomy (in hours)")
    total_degrees_of_freedom_dof: Optional[int] = Field(None, alias="Total Degrees of Freedom (DOF)")
    body_dof: Optional[int] = Field(None, alias="Body DOF")
    hands_dof: Optional[int] = Field(None, alias="Hands DOF")
    two_hand_payload_kg: Optional[float] = Field(None, alias="Two-hand Payload (in kg)")
    vision_sensors_type: Optional[str] = Field(None, alias="Vision Sensors type")
    speaker: Optional[str] = Field(None, alias="Speaker")
    microphone: Optional[str] = Field(None, alias="Microphone")
    safety_features: Optional[str] = Field(None, alias="Safety Features")
    actuator_type: Optional[str] = Field(None, alias="Actuator Type")
    reducer_type: Optional[str] = Field(None, alias="Reducer Type")
    motor_type: Optional[str] = Field(None, alias="Motor Type")
    force_sensor: Optional[str] = Field(None, alias="Force Sensor")
    encoder_per_actuator: Optional[int] = Field(None, alias="Encoder per Actuator")
    computing_power: Optional[str] = Field(None, alias="Computing Power")
    ai_technology_used: Optional[str] = Field(None, alias="AI Technology used")
    ai_partners: Optional[str] = Field(None, alias="AI Partners")
    customers_or_testers: Optional[str] = Field(None, alias="Customers or Testers")
    can_converse_naturally: Optional[str] = Field(None, alias="Can the robot converse naturally?")

    def to_csv_row(self) -> str:
        output = StringIO()
        writer = csv.writer(output)

        # Get the alias names in the order of the model's fields
        field_names = [field.alias for field in self.__pydantic_fields__.values()]
        values = [getattr(self, name) for name in self.__pydantic_fields__.keys()]

        # Write single row
        writer.writerow(values)
        return output.getvalue().strip()

    class Config:
        validate_by_name = True


# Computation of the percentage of filled fields for a given robot

def completion_rate(robot: RobotSpec) -> float:
    """
    Computes the completion percentage of a RobotSpec instance.
    Fields with value `None` or string 'n/d' are considered missing.
    """
    values = robot.model_dump()
    total = len(values)

    def is_filled(val):
        if val is None:
            return False
        if isinstance(val, str) and val.strip().lower() == "n/d":
            return False
        return True

    filled = sum(1 for val in values.values() if is_filled(val))
    return round((filled / total) * 100, 2)


def get_empty_fields(robot: RobotSpec) -> List[str]:
    """
    Returns the list of empty fields (None or 'n/d') in a RobotSpec instance,
    using field aliases where defined.
    """
    empty_fields = []

    for field_name, value in robot.model_dump().items():
        is_empty = value is None or (isinstance(value, str) and value.strip().lower() == "n/d")
        if is_empty:
            alias = RobotSpec.model_fields[field_name].alias or field_name
            empty_fields.append(alias)

    return empty_fields

def complete_robot_fields(robot: RobotSpec, update_data: dict) -> List[str]:
    updated_fields = []
    model_fields = robot.__class__.model_fields  # Pydantic v2 syntax

    for alias, value in update_data.items():
        if value in [None, -1, "n/d", "n/a", "N/A", "unknown", "no data", ""]:
            continue

        for internal_name, field_info in model_fields.items():
            if field_info.alias == alias:
                setattr(robot, internal_name, value)
                updated_fields.append(internal_name)
                break

    return updated_fields


os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

search_tavily = TavilySearch()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search = Tool(
        name="Search",
        func=GoogleSerperAPIWrapper().run,
        description="Use this tool to search the web for information about robots."
    )

from langchain_core.prompts import ChatPromptTemplate

llm_retrieves = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
)

retrieving_prompt = "You are a detective finding if a humanoid robot is referenced on a given website. You have access to a search tool."

retrieving_llm = create_react_agent(llm_retrieves,[search_tavily], prompt=retrieving_prompt)

from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        robot: current states of completion of robot's specs
        completion_rate: % of completion of robot's specs
        remaining_fields : fields not yet completed for the robot
        prompt : prompt previously given to webscraper robot
    """
    robot: RobotSpec
    completion_rate: float
    remaining_fields: List[str]
    prompt : str
    url : str

import requests
from bs4 import BeautifulSoup

@tool
def fetch_full_webpage(url: str) -> str:
    """Downloads and returns the readable text content of a given webpage."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        return f"Error fetching page: {e}"


llm_webscrape = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt_webscrape = """ You are very powerful assistant, helping the user fill a database of humanoid robots. 
            The user will give you the name of a robot and a url and you will have access to a tool to output the following information :
            You have access to a tool called fetch_full_webpage that takes a URL and returns the full readable text content of the webpage.
When you need info from the page, call the tool with: fetch_full_webpage("<url>")
Then analyze the returned text and extract the requested data.
            IMPORTANT RULES:
    - If you cannot find specific values, write `n/d` exactly.
    - Use ONLY the content found in the URLs. DO NOT assume or invent any data.
    - If a data point is missing, write exactly `n/d` for string fields and -1 for numeric ones (no other variation).
    - For yes/no fields, respond strictly with `yes` or `no`.
    - For numeric fields, return only the number (no units, no text).
      - No extra text or formatting (no bullet points, no explanations, no blank lines).
      - Multiple elements for a same field must be separated by a semi-colon ";" and written between []


    💡 Each value must follow the format described below:

    | Field Name                                | Type          | Format / Notes                                                                 |
    |------------------------------------------ |---------------|--------------------------------------------------------------------------------|
    | Robot Name                                | text          | Full robot name                                                                |
    | Company                                   | text          | Company name                                                                   |
    | Country                                   | text          | Country where the company is based                                             |
    | Region                                    | text          | Continent or macro-region (e.g., North America, Europe)                        |
    | Year Unveiled                             | number        | Just the year the robot was unveiled for the first time  (e.g., 2023)                                                     |
    | Primary Use-Case                          | single choice | Choose from: Industrial, Research, Industrial/Logistics, General Purpose, Service |
    | Status                                    | single choice | Prototype or In Production                                                     |
    | Price (USD)                                | number or -1 | No $ sign or commas                                                            |
    | Production Capacity (units per year)      | number or n/d | Integer only                                                                   |
    | Height (in cm)                            | number        | Centimeters only                                                               |
    | Weight (in kg)                            | number        | Kilograms only                                                                 |
    | Mobility Type                             | text          | e.g., bipedal, wheeled, modular                                                |
    | Speed (in m/s)                            | number        | Meters per second                                                              |
    | Autonomy (in hours)                       | number or n/d | Duration the robot can operate without recharge                                |
    | Total Degrees of Freedom (DOF)            | number        | Integer only                                                                   |
    | Body DOF                                  | number        | Integer only                                                                   |
    | Hands DOF                                 | number        | Integer only                                                                   |
    | Two-hand Payload (in kg)                  | number        | Integer or decimal                                                             |
    | Vision Sensors type                       | text          | e.g., RGB-D, LiDAR, Fisheye, RGB, ...                                          |
    | Speaker                                   | yes/no        | Only `yes` or `no`                                                             |
    | Microphone                                | yes/no        | Only `yes` or `no`                                                             |
    | Safety Features                           | text or n/d   | Short description or `n/d`                                                     |
    | Actuator Type                             | text or n/d   | e.g., electric, hydraulic                                                      |
    | Reducer Type                              | text or n/d   | e.g., Planetary, Harmonic, RV                                                  |
    | Motor Type                                | text or n/d   | e.g., BLDC, AC                                                                 |
    | Force Sensor                              | yes/no        | Only `yes` or `no`                                                             |
    | Encoder per Actuator                      | number or n/d | Integer only                                                                   |
    | Computing Power                           | text or n/d   | e.g., NVIDIA Orin, 275 TOPS AI chip,...                                        |
    | AI Technology used                        | text or n/d          | e.g., LLM; CV; RL — semi-colon separated                                          |
    | AI Partners                               | text or n/d   | Company or lab names                                                           |
    | Customers or Testers                      | text or n/d   | Company names or `n/d`                                                         |
    | Can the robot converse naturally?         | yes/no        | Only `yes` or `no`                                                             |


    Your output must be in **JSON format only**, exactly in the same order as the fields were presented to you.
    USE THE EXACT ALIASES.
    Be extremely strict with formatting. No markdown, no additional characters, no comments.
"""

web_scraper = create_react_agent(llm_webscrape, [fetch_full_webpage], prompt=prompt_webscrape)

import json
import re


def extract_json_part(text: str) -> dict:
    pattern = r'\{(?:[^{}]|(?R))*\}'

    start = text.find('{')
    if start == -1:
        raise ValueError("Error parsing")

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = text[start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON bad format: {e}")
    raise ValueError("JSON not complete")


llm_websearch = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [search_tavily, fetch_full_webpage]

prompt_websearch = """You are very powerful assistant, helping the user fill a database of humanoid robots. 
            You are allowed to perform **multiple searches and tool uses** before answering. Use the tools as many times as necessary to gather all data.
            The user will give you the name of a robot and you will have to output the following information :

            IMPORTANT RULES:
    - If the tool fails with a ToolException (e.g. "TAVILY_LIMIT_REACHED"), you must produce the most accurate JSON file possible based only on what you already know or deduced from prior tool results.
    - If you cannot find specific values, write `n/d` exactly.
    - Use ONLY the content found in the URLs. DO NOT assume or invent any data.
    - If a data point is missing, write exactly `n/d` for string fields and -1 for numeric ones (no other variation).
    - For yes/no fields, respond strictly with `yes` or `no`.
    - For numeric fields, return only the number (no units, no text).
      - No extra text or formatting (no bullet points, no explanations, no blank lines).
      - Multiple elements for a same field must be separated by a semi-colon ";" and written between []

    💡 Each value must follow the format described below:

    | Field Name                                | Type          | Format / Notes                                                                 |
    |------------------------------------------ |---------------|--------------------------------------------------------------------------------|
    | Robot Name                                | text          | Full robot name                                                                |
    | Company                                   | text          | Company name                                                                   |
    | Country                                   | text          | Country where the company is based (United States should be written in full letters as United Kingdom) |
    | Region                                    | text          | Continent or macro-region (e.g., North America, Europe)                        |
    | Year Unveiled                             | number        | Just the year the robot was unveiled for the first time  (e.g., 2023)          |
    | Primary Use-Case                          | single choice | Choose from: Industrial, Research, Industrial/Logistics, General Purpose, Service |
    | Status                                    | single choice | Prototype or In Production                                                     |
    | Price (USD)                                | number  | No $ sign or commas                                                            |
    | Production Capacity (units per year)      | number or n/d | Integer only                                                                   |
    | Height (in cm)                            | number        | Centimeters only                                                               |
    | Weight (in kg)                            | number        | Kilograms only                                                                 |
    | Mobility Type                             | text  or n/d        | e.g., bipedal, wheeled, modular                                                |
    | Speed (in m/s)                            | number        | Meters per second                                                              |
    | Autonomy (in hours)                       | number  | Duration the robot can operate without recharge                                |
    | Total Degrees of Freedom (DOF)            | number        | Integer only                                                                   |
    | Body DOF                                  | number        | Integer only                                                                   |
    | Hands DOF                                 | number        | Integer only                                                                   |
    | Two-hand Payload (in kg)                  | number        | Integer or decimal                                                             |
    | Vision Sensors type                       | text          | e.g., RGB-D, LiDAR, Fisheye, RGB, ...                                          |
    | Speaker                                   | yes/no        | Only `yes` or `no`                                                             |
    | Microphone                                | yes/no        | Only `yes` or `no`                                                             |
    | Safety Features                           | text or n/d   | Short description or `n/d`                                                     |
    | Actuator Type                             | text or n/d   | e.g., electric, hydraulic                                                      |
    | Reducer Type                              | text or n/d   | e.g., Planetary, Harmonic, RV                                                  |
    | Motor Type                                | text or n/d   | e.g., BLDC, AC                                                                 |
    | Force Sensor                              | yes/no        | Only `yes` or `no`                                                             |
    | Encoder per Actuator                      | number or n/d | Integer only                                                                   |
    | Computing Power                           | text or n/d   | e.g., NVIDIA Orin, 275 TOPS AI chip,...                                        |
    | AI Technology used                        | text or n/d        | e.g., LLM; CV; RL — semi-colon separated                                          |
    | AI Partners                               | text or n/d   | Company or lab names                                                           |
    | Customers or Testers                      | text or n/d   | Company names or `n/d`                                                         |
    | Can the robot converse naturally?         | yes/no        | Only `yes` or `no`                                                             |

    Your output must be in **JSON format only**, exactly in the same order as the fields were presented to you.
    USE THE EXACT ALIAS.
    Be extremely strict with formatting. No markdown, no additional characters, no comments.
"""

web_search_worker = create_react_agent(llm_websearch, tools, prompt=prompt_websearch)


def humanoid_guided_websearch(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates robots' specs from scrape results
    """

    print("---WEB SCRAPE---")
    robot = state["robot"]
    url = state["url"]

    print(robot.robot_name, robot.company, url)

    found_url = retrieving_llm.invoke({"messages": [("human", f"""
Search the web using the following format and using the tools you have:
{url} {robot.robot_name} {robot.company}

Determine if there exists a specific page that contains information about the humanoid robot mentioned.

✅ If such a page exists, return ONLY the full URL of that exact page (not the homepage).

❌ Do NOT include any explanations, summaries, or comments.

❌ Do NOT respond if you are uncertain.

If no relevant page is found, respond with exactly:
NOT FOUND
""")]})["messages"][-1].content

    print(found_url)
    if found_url != "NOT FOUND":
        # Web search
        scrape_result = web_scraper.invoke({
            "messages": ("human",
                         "Find data about" + robot.robot_name + " " + robot.company + "humanoid robot. Use the following URL: " + found_url)})[
            "messages"][-1].content
        complete_robot_fields(robot, extract_json_part(scrape_result))
    return {
        **state,
        "remaining_fields": get_empty_fields(robot),
        "completion_rate": completion_rate(robot),
        "prompt": prompt_webscrape,
        "url": "www.aparobot.com"
    }


def general_websearch(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates robots' specs from web results
    """

    print("---WEB SEARCH---")
    robot = state["robot"]
    prompt = prompt_websearch

    # Web search
    search_result = web_search_worker.invoke({
        "messages": ("human", "Find data about" + robot.robot_name + " " + robot.company + "humanoid robot.")})[
        "messages"][-1].content

    complete_robot_fields(robot, extract_json_part(search_result))
    return {
        **state,
        "remaining_fields": get_empty_fields(robot),
        "completion_rate": completion_rate(robot),
        "prompt": prompt,
    }


def checking_condition_1(state):
    if state["completion_rate"] >= 80:
        return END
    else:
        return "websearch_url2"


def checking_condition_2(state):
    if state["completion_rate"] >= 80:
        return END
    else:
        return "general_web_search"

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Defining nodes

workflow.add_node("websearch_url1", humanoid_guided_websearch)

workflow.set_entry_point("websearch_url1")

workflow.add_node("websearch_url2", humanoid_guided_websearch)

workflow.add_conditional_edges("websearch_url1",checking_condition_1)
workflow.add_conditional_edges("websearch_url2",checking_condition_2)

workflow.add_node("general_web_search",general_websearch)
workflow.add_edge("general_web_search",END)

app = workflow.compile()

llm_find_robots = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
)

finding_robots_prompt = """
You are a detective specialized in finding all the latest humanoid robots unveiled on the following websites:
- www.aparobot.com
- humanoidroboticstechnology.com
- www.linkedin.com
- x.com

IMPORTANT DEFINITIONS AND RULES:
- A **humanoid robot** is a robot with a human-like form: typically bipedal, or wheeled with a torso, arms, and/or head.
- EXCLUDE ALL animal-inspired robots such as robot dogs, robot cats, quadrupeds, insect-like robots, snake robots, or any zoomorphic robots.
- DO NOT include ANY robot that walks on four legs or resembles animals in shape or behavior.

Use the available tools to search and extract at least 30 recent humanoid robots.
"""

find_robots_agent = create_react_agent(llm_find_robots,[search, fetch_full_webpage], prompt=finding_robots_prompt)

latest_robots=find_robots_agent.invoke({"messages":[("human",f"""
Using the available tools as many times as you want, find the most recent **humanoid** robots (bipedal or wheeled ONLY — absolutely NO quadrupeds or animal-like robots) unveiled on the given websites.

IMPORTANT OUTPUT RULES:
- Your output MUST be a single, valid JSON array.
- Each element must be a JSON object with exactly these keys:
    - "name" (the robot's name),
    - "company" (the company that unveiled it)
- DO NOT include any explanations, bullet points, or extra formatting — ONLY valid JSON.

Example format:
[
  {{"name": "RobotX", "company": "FutureCorp"}},
  {{"name": "RobotY", "company": "TechnoBots"}}
]
""")]})["messages"][-1].content
print(latest_robots)

import json
import re

def parse_robot_json_output(llm_output: str):
    try:
        # Extraction brutale du JSON s'il est mélangé avec d'autres textes
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in the output.")

        json_str = match.group(0)

        data = json.loads(json_str)

        # Validation minimale
        if not isinstance(data, list):
            raise ValueError("Parsed JSON is not a list.")
        for robot in data:
            if not isinstance(robot, dict) or \
               not all(k in robot for k in ("name", "company")):
                raise ValueError("Invalid robot entry found.")

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

from io import StringIO
import os

import time

cpt_count = 0

latest_robots =parse_robot_json_output(latest_robots)
def get_existing_line_count(filename, has_header=True):
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return len(lines) - (1 if has_header else 0)


def process_robots(json_list, filename, start_index=0):
    all_res = []
    nb_lines = get_existing_line_count(filename)

    for cpt_count, element in enumerate(json_list[start_index:], start=start_index):
        tp_robot = RobotSpec(robot_name=element["name"], company=element["company"])
        rem_fields = get_empty_fields(tp_robot)
        initial_state = {
            "robot": tp_robot,
            "completion_rate": completion_rate(tp_robot),
            "remaining_fields": rem_fields,
            "prompt": "",
            "url": "humanoid.guide"
        }

        try:
            for output in app.stream(initial_state):
                for key, value in output.items():
                    pass
        except Exception as e:
            print(f"\n⚠️  Error caught at robot #{cpt_count}: {tp_robot.robot_name}")
            print(f"Exception: {e}")
            choice = input("❓ Do you want to continue from this point later? (y/n): ").lower()
            if choice == "y":
                print(f"\n👉 You can restart the script from `start_index={cpt_count}`.")
                return all_res  # Return partial result for reuse
            else:
                print("❌ Aborting processing.")
                break

        robot = value["robot"]
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(str(cpt_count + nb_lines) + ", " + robot.to_csv_row() + "\n")

        all_res.append(robot)
        time.sleep(30)

    return all_res

complete_list=process_robots(latest_robots,"../../humanoid_data_testing_2207.csv")

import pandas as pd
df_string="Robot Name,Company,Country,Region,Year Unveiled,Primary Use-Case,Status,Cost (USD),Production Capacity (units per year),Height (in cm),Weight (in kg),Mobility Type,Speed (in m/s),Autonomy (in hours),Total Degrees of Freedom (DOF),Body DOF,Hands DOF,Two-hand Payload (in kg),Vision Sensors type,Speaker,Microphone,Safety Features,Actuator Type,Reducer Type,Motor Type,Force Sensor,Encoder per Actuator,Computing Power,AI Technology used,AI Partners,Customers or Testers,Can the robot converse naturally?"+"\n"
for robot in complete_list :
    df_string+=robot.to_csv_row()
    df_string+="\n"
df_humanoid_data= pd.read_csv(StringIO(df_string))
df_sans_doublons = df_humanoid_data.drop_duplicates(subset=['Robot Name'], keep='last')

# Sauvegarder dans un nouveau fichier
#df_sans_doublons.to_csv('humanoid_data_testing_2207.csv', index=False)

import os
import ast
from langchain.chat_models import ChatOpenAI

# Clé OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# corriger hdr c'est quoi ?
# Prompt template
prompt_template = """
You are a data cleaning assistant. Clean and standardize the values in the following column according to these rules:
    -Unify similar terms: Map all semantically equivalent or similar terms to a consistent, standard format.
        Example: "deep learning", "DeepLearning", "DL" → "DL"
        Example: "RGB-D", "3D", "Depth camera", "stereo" → "RGBD"
        Example : "Advanced safety systems", "Advanced safety", "Comprehensive safety...", and similar -> "Advanced safety protocols"
    -Remove generic or vague values:
        Discard entries that are too broad, ambiguous, or meaningless in context (e.g., "AI", "framework", "open-source").
    -Keep only specific, relevant technologies or components:
        For instance, valid values might include AI models like "YOLOv5", "GPT-4","ML" is useless, better to list techniques like "DL", "RL", "VLA" or others, or concrete hardware/software like "NVIDIA Jetson", "LiDAR", "RGBD".
    -Split compound entries:
        Replace separators like "and", "&", "/" with a semi-colon (";") only if all terms are individually valid and relevant.
    -Otherwise, remove the entry entirely.
    -Discard incoherent, mixed, or unrelated values:
        If a value combines unrelated concepts (e.g., "AI and BigData" in the "Computing Power" column), return an empty string.
    -When in doubt, remove the value.
Output format:
    - Return a Python dictionary where each original value (string) is a key, and the cleaned value (or empty string) is the corresponding value.
Column: {column_name}  
Values: {column_values}
"""

# LLM setup
llm = ChatOpenAI(temperature=0, model="gpt-4")


def clean_and_replace_column(df, col):
    try:
        unique_values = df[col].dropna().astype(str).unique().tolist()
        prompt = prompt_template.format(column_name=col, column_values=unique_values)
        response_text = llm.predict(prompt)

        cleaned_mapping = ast.literal_eval(response_text)

        df[col] = df[col].astype(str).map(cleaned_mapping).fillna("")
        return response_text
    except Exception as e:
        return f"An error occurred: {str(e)}"


col_to_check = ["Vision Sensors type", "Speaker", "Microphone", "Safety Features", "Actuator Type", "Reducer Type",
                "Motor Type", "Force Sensor", "Encoder per Actuator", "Computing Power", "AI Technology used"]

for col in col_to_check:
    clean_and_replace_column(df_sans_doublons, col)
    print(col + " cleaned")

print(df_sans_doublons)
df_sans_doublons.to_csv("../../data/humanoid_data_cleaned2207.csv")
