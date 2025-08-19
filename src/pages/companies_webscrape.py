from dotenv import load_dotenv
from langchain.agents import Tool
import pandas as pd
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import io, csv, re, os, json
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()


def companies_webscrape(general_directory):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

    search = Tool(
        name="Search",
        func=TavilySearch(),
        description="Use this tool to search the web for information about robots."
    )

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

    prompt_webscrape_company = """
    You are a powerful assistant tasked with extracting financial and operational information about companies that produce humanoid robots.

    You are allowed to perform MULTIPLE web searches and use any URL content to extract the data. Use the tools available as many times as needed.

    You have access to:
    - search(query): performs a web search and gives URLs
    - fetch_full_webpage("<url>"): fetches the readable text of any page

    You MUST use both tools as needed to find and extract the most accurate values.

    IMPORTANT RULES:
    - Only use content found in the webpages. Do NOT infer, guess or hallucinate.
    - If a value is missing or unavailable, write exactly:
      - `n/d` for text fields
      - `-1` for numeric fields
    - Dates: use year only (e.g. 2015)
    - If multiple values apply to one field, separate by `;` and enclose in brackets: e.g., `[Company A; Company B]`
    - Numeric fields: must contain ONLY the number, no text, no units, no currency symbols
    - Use trusted sources: Google Finance, Crunchbase, Pitchbook, Tracxn, company websites, etc.
    - Summarize values if needed (e.g. total funding from multiple rounds)

    FIELDS TO EXTRACT:
    Return your output in JSON ONLY, with keys and order EXACTLY as below:

    {
      "Company Name": "",
      "Most important robot": "",
      "Total Funding (USD)": -1,
      "Market Capitalization (USD) or Valuation": -1,
      "Number of Employees": -1,
      "Founding Date": "n/d",
      "Planned Robot Production Units": -1,
      "Partner Companies in automobile world": "n/d"
    }
    For market capitalization it can be the valuation of the company. It should be the capitalization or valuation to this date not future one !
    For Total Funding you must give the sum of all the money the company raised since it was created to this date !
    Just write the corresponding data and do not add any information other than what is asked !
    """


    llm_webscrape = ChatOpenAI(model="gpt-4o-mini",
                               temperature=0,
                               openai_api_key=os.getenv("OPENAI_API_KEY"))

    web_scraper = create_react_agent(llm_webscrape,
                                     tools=[search, fetch_full_webpage],
                                     prompt=prompt_webscrape_company)

    recap_companies = []

    def extract_json(text):
        match_block = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match_block:
            try:
                return json.loads(match_block.group(1))
            except json.JSONDecodeError:
                pass

        match_string = re.search(r"({\\n.*?})", text)
        if match_string:
            try:
                cleaned = match_string.group(1).encode('utf-8').decode('unicode_escape')
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        match_brut = re.search(r"({[^{}]*})", text)
        if match_brut:
            try:
                return json.loads(match_brut.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError("Json not found.")

    def to_csv_row(data: dict, fieldnames: list, delimiter=","):

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
        writer.writerow(data)
        return output.getvalue().strip("\r\n")


    df_robots = pd.read_csv(os.path.join(general_directory,"../../data/humanoid_data_cleaned.csv"))
    companies = df_robots["Company"].unique()

    filename=os.path.join(general_directory,"../../companies_data.csv")
    columns=["Company Name","Most important robot","Total Funding (USD)","Market Capitalization (USD) or Valuation","Number of Employees","Founding Date","Planned Robot Production Units","Partner Companies in automobile world"]

    for company in companies:
        try :
            tp = web_scraper.invoke({
                "messages": [
                    ("user", "Find data about " + company)
                ]
            })["messages"][-1].content
            element_found = extract_json(tp)
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(to_csv_row(element_found, columns) + "\n")

        except Exception as e:
            print(f"Error with {company}: {e}")
            pass
















