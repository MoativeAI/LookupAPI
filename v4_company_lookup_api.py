from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from agent_tools.web_page_tool import WebPageTool
from agent_tools.link_retrieval_tool import LinkRetriever
from agent_tools.metadesc_tool import MetaDescriptionTool
import naics_rag.query
import time

import csv

load_dotenv()
app = FastAPI()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
)

class CompanyInfo(BaseModel):
    Company_URL: str
    Company_LinkedIn_URL: str
    Company_Facebook_URL: str
    Company_Twitter_URL: str
    Company_Phone: str
    Company_Address: str
    Meta_Description: str
    Overview: str
    USP: str
    Target_Audience: str
    Conclusion: str
    NAICS_Code: str
    Title: str
    Description: str
    Common_Labels: str
    execution_time: float

class PrimaryCompanyInfo(BaseModel):
    Company_URL: str
    Meta_Description: str
    Overview: str
    Industry_Type: str
    execution_time: float

def create_default_company_info(execution_time: float) -> CompanyInfo:
    return CompanyInfo(
        Company_URL="NOT_FOUND",
        Company_LinkedIn_URL="NOT_FOUND",
        Company_Facebook_URL="NOT_FOUND",
        Company_Twitter_URL="NOT_FOUND",
        Company_Phone="NOT_FOUND",
        Company_Address="NOT_FOUND",
        Meta_Description="NOT_FOUND",
        Overview="NOT_FOUND",
        USP="NOT_FOUND",
        Target_Audience="NOT_FOUND",
        Conclusion="NOT_FOUND",
        NAICS_Code="NOT_FOUND",
        Title="NOT_FOUND",
        Description="NOT_FOUND",
        Common_Labels="NOT_FOUND",
        execution_time=execution_time
    )

def create_default_primary_company_info(execution_time: float) -> PrimaryCompanyInfo:
    return PrimaryCompanyInfo(
        Company_URL="NOT_FOUND",
        Meta_Description="NOT_FOUND",
        Overview="NOT_FOUND",
        Industry_Type="NOT_FOUND",
        execution_time=execution_time
    )

meta_description_tool = MetaDescriptionTool()
page_getter = WebPageTool()
google_search_tool = LinkRetriever(
    api_key=os.environ["GOOGLE_API_KEY"],
    cse_id=os.environ["GOOGLE_CSE_ID"],
)

@app.get("/lookup/company/{company_name}", response_model=CompanyInfo)
async def lookup_company(company_name: str):
    start_time = time.time()
    def analyze_company(company_name):
        combined_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a comprehensive company information retriever. Your purpose is to provide the following information for the company name given as input:\n"
             "1. Official website URL\n"
             "2. LinkedIn URL\n"
             "3. Facebook URL\n"
             "4. Twitter URL\n"
             "Use the google_search_tool to find the URLs."
             "Respond in the following format:\n\n"
             "Company_URL: [insert URL here]\n"
             "Company_LinkedIn_URL: [insert URL here]\n"
             "Company_Facebook_URL: [insert URL here]\n"
             "Company_Twitter_URL: [insert URL here]\n"
             "If you cannot find definitive information for any item, respond with 'NOT_FOUND' for that specific item."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        start_links_time = time.time()
        tools = [google_search_tool]
        website_agent = create_tool_calling_agent(llm, tools, combined_prompt)
        website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)
        website_links = website_agent_executor.invoke({"input": company_name})
        output = website_links['output']
        end_links_time = time.time()
        links_time = end_links_time-start_links_time

        lines = output.strip().split('\n')
        company_dict = {}
        for line in lines[:1]:
            key, value = line.split(': ', 1)
            company_dict[key] = value.strip()

        url = company_dict['Company_URL']
        if url != "NOT_FOUND":
            combined_company_analysis_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a comprehensive company analyzer. Your purpose is to provide detailed information about a given company based on its website content. Use the meta_description_tool to fetch the meta description and the page_getter tool for other company information. Respond in the following format:\n\n"
                 "Meta_Description: [insert meta description here]\n"
                 "Company_Phone: [insert phone number here]\n"
                 "Company_Address: [insert address here]\n"
                 "Overview: [insert overview here]\n"
                 "USP: [insert unique selling proposition here]\n"
                 "Target_Audience: [insert target audience here]\n"
                 "Conclusion: [insert conclusion here]\n\n"
                 "If you cannot find enough information for any field, respond with 'Information not available' for that specific field. Don't give me any other text"),
                ("placeholder", "{chat_history}"),
                ("human", "Analyze this company: {url}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            NAICS_PROMPT_TEMPLATE = """You are an expert in NAICS (North American Industry Classification System) codes. Your task is to analyze the given company description and determine the most appropriate NAICS code based on the information provided in the context below. The context contains relevant excerpts from the NAICS manual.

            Context from NAICS manual:
            {context}

            ---

            Company Description: {question}

            Based on the company description and the NAICS information provided in the context, please determine the most appropriate NAICS code for this company.

            Please format your response as follows:
            NAICS_Code: [code]
            Title: [title of the code]
            Description: [brief description]
            Common_Labels: [labels that are more commonly used than the title of the code]
            """
            start_company_time = time.time()
            company_tools = [meta_description_tool, page_getter]
            company_agent = create_tool_calling_agent(llm, company_tools, combined_company_analysis_prompt)
            company_agent_executor = AgentExecutor(agent=company_agent, tools=company_tools, verbose=True)
            company_result = company_agent_executor.invoke({"url": url})
            company_info = company_result['output']
            end_company_time = time.time()
            company_time = end_company_time-start_company_time

            start_naics_time = time.time()
            results = naics_rag.query.query_rag(company_info, NAICS_PROMPT_TEMPLATE)
            end_naics_time = time.time()
            naics_time = end_naics_time-start_naics_time

            text = output + '\n' + company_info + '\n' + results.content
            lines = text.strip().split('\n')
            data_dict = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(":", 1)
                    data_dict[key.strip()] = value.strip()

            end_time = time.time()
            execution_time = end_time - start_time
            data_dict['execution_time'] = execution_time
            # filename = "v4_company_lookups.csv"
            # file_exists = os.path.isfile(filename)
            #
            # with open(filename, mode='a', newline='') as file:
            #     writer = csv.DictWriter(file, fieldnames=data_dict.keys())
            #
            #     if not file_exists:
            #         writer.writeheader()
            #
            #     writer.writerow(data_dict)

            output_file = 'combined_timing_data.csv'
            with open(output_file, 'a', newline='') as out_csv:
                writer = csv.writer(out_csv)

                if out_csv.tell() == 0:
                    writer.writerow(['company_name', 'links_time', 'company_time', 'naics_time', 'execution_time'])

                writer.writerow([company_name, links_time, company_time, naics_time, execution_time])
            return CompanyInfo(**data_dict)
        else:
            end_time = time.time()
            execution_time = end_time - start_time
            return create_default_company_info(execution_time)

    return analyze_company(company_name)

@app.get("/lookup/company/primary/{company_name}", response_model=PrimaryCompanyInfo)
async def lookup_company(company_name: str):
    start_time = time.time()
    def primary_analyze_company(company_name):
        combined_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a comprehensive company information retriever. Your purpose is to provide the following information for the company name given as input:\n"
             "1. Official website URL\n"
             "Use the google_search_tool to find the URL."
             "Respond in the following format:\n\n"
             "Company_URL: [insert URL here]\n"
             "If you cannot find definitive information for any item, respond with 'NOT_FOUND' for that specific item."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        tools = [google_search_tool]
        website_agent = create_tool_calling_agent(llm, tools, combined_prompt)
        website_agent_executor = AgentExecutor(agent=website_agent, tools=tools, verbose=True)
        website_links = website_agent_executor.invoke({"input": company_name})
        output = website_links['output']
        key, value = output.split(': ', 1)
        url = value.strip()
        if url != "NOT_FOUND":
            combined_company_analysis_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a comprehensive company analyzer. Your purpose is to provide detailed information about a given company based on its website content. Use the meta_description_tool to fetch the meta description and the page_getter tool for other company information. Respond in the following format:\n\n"
                 "Meta_Description: [insert meta description here]\n"
                 "Overview: [insert overview here]\n"
                 "If you cannot find enough information for any field, respond with 'Information not available' for that specific field. Don't give me any other text"),
                ("placeholder", "{chat_history}"),
                ("human", "Analyze this company: {url}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            NAICS_PROMPT_TEMPLATE = """You are an expert in NAICS (North American Industry Classification System) codes. Your task is to analyze the given company description and determine the most appropriate NAICS code based on the information provided in the context below. The context contains relevant excerpts from the NAICS manual.

            Context from NAICS manual:
            {context}

            ---
            
            Company Description: {question}

            Based on the company description and the NAICS information provided in the context, please determine the most appropriate NAICS code for this company.

            Please retrieve the NAICS Code and use it to format your response as follows:
            Industry_Type: [title of the code, and labels that are more commonly used than the title of the code]
            """

            company_tools = [meta_description_tool, page_getter]
            company_agent = create_tool_calling_agent(llm, company_tools, combined_company_analysis_prompt)
            company_agent_executor = AgentExecutor(agent=company_agent, tools=company_tools, verbose=True)
            company_result = company_agent_executor.invoke({"url": url})
            company_info = company_result['output']
            results = naics_rag.query.query_rag(company_info, NAICS_PROMPT_TEMPLATE)
            text = output + '\n' + company_info + '\n' + results.content
            lines = text.strip().split('\n')
            data_dict = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(":", 1)
                    data_dict[key.strip()] = value.strip()

            end_time = time.time()
            execution_time = end_time - start_time
            data_dict['execution_time'] = execution_time
            return PrimaryCompanyInfo(**data_dict)
        else:
            end_time = time.time()
            execution_time = end_time - start_time
            return create_default_primary_company_info(execution_time)

    return primary_analyze_company(company_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)