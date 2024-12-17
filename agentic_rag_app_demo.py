#! /usr/bin/env python

import os
import gradio as gr
import requests
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai_tools import tool
from crewai import Crew, Task, Agent

# API Key Configuration 
os.environ['GROQ_API_KEY'] =  'YOUR_GROQ_API_KEY'
os.environ['TAVILY_API_KEY'] = 'YOUR_TAVILY_API_KEY'

# Download PDF if not already present
def download_pdf():
    pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
    if not os.path.exists('attention_is_all_you_need.pdf'):
        response = requests.get(pdf_url)
        with open('attention_is_all_you_need.pdf', 'wb') as file:
            file.write(response.content)

# LLM Configuration
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000,
)

# Tools Configuration
def setup_tools():
    # RAG Tool
    rag_tool = PDFSearchTool(
        pdf='attention_is_all_you_need.pdf',
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-8b-8192",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

    # Web Search Tool
    web_search_tool = TavilySearchResults(k=3)

    return rag_tool, web_search_tool

# Router Tool
@tool
def router_tool(question):
    """Router Function"""
    keywords = ['self-attention', 'transformer', 'attention', 'language model']
    if any(keyword in question.lower() for keyword in keywords):
        return 'vectorstore'
    return 'web_search'

# Agent Definitions
def create_agents():
    Router_Agent = Agent(
        role='Router',
        goal='Route user question to a vectorstore or web search',
        backstory=(
            "You are an expert at routing a user question to a vectorstore or web search. "
            "Use the vectorstore for questions related to Retrieval-Augmented Generation. "
            "Be flexible in interpreting keywords related to these topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Retriever_Agent = Agent(
        role="Retriever",
        goal="Use retrieved information to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks. "
            "Provide clear, concise answers using retrieved context."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Grader_agent = Agent(
        role='Answer Grader',
        goal='Filter out irrelevant retrievals',
        backstory=(
            "You are a grader assessing the relevance of retrieved documents. "
            "Evaluate if the document contains keywords related to the user question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Hallucination_Grader = Agent(
        role="Hallucination Grader",
        goal="Verify answer factuality",
        backstory=(
            "You are responsible for ensuring the answer is grounded in facts "
            "and directly addresses the user's question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Final_Answer_Agent = Agent(
        role="Final Answer Agent",
        goal="Provide a comprehensive and accurate response",
        backstory=(
            "You synthesize information from various sources to create "
            "a clear, concise, and informative answer to the user's question."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    return Router_Agent, Retriever_Agent, Grader_agent, Hallucination_Grader, Final_Answer_Agent

# Task Definitions
def create_tasks(agents, tools):
    rag_tool, web_search_tool = tools
    Router_Agent, Retriever_Agent, Grader_agent, Hallucination_Grader, Final_Answer_Agent = agents

    router_task = Task(
        description=(
            "Analyze the keywords in the question {question}. "
            "Decide whether it requires a vectorstore search or web search."
        ),
        expected_output="Return 'websearch' or 'vectorstore'",
        agent=Router_Agent,
        tools=[router_tool],
    )

    retriever_task = Task(
        description=(
            "Retrieve information for the question {question} "
            "using either web search or vectorstore based on router task."
        ),
        expected_output="Provide retrieved information",
        agent=Retriever_Agent,
        context=[router_task],
        tools=[rag_tool, web_search_tool],
    )

    grader_task = Task(
        description="Evaluate the relevance of retrieved content for the question {question}",
        expected_output="Return 'yes' or 'no' for relevance",
        agent=Grader_agent,
        context=[retriever_task],
    )

    hallucination_task = Task(
        description="Verify if the retrieved answer is factually grounded",
        expected_output="Return 'yes' or 'no' for factual accuracy",
        agent=Hallucination_Grader,
        context=[grader_task],
    )

    answer_task = Task(
        description=(
            "Generate a final answer based on retrieved and verified information. "
            "Perform additional search if needed."
        ),
        expected_output="Provide a clear, concise answer",
        agent=Final_Answer_Agent,
        context=[hallucination_task],
        tools=[web_search_tool],
    )

    return [router_task, retriever_task, grader_task, hallucination_task, answer_task]

# Main RAG Function
def run_rag_pipeline(question):
    # Download PDF if not exists
    download_pdf()

    # Setup tools
    rag_tool, web_search_tool = setup_tools()
    tools = (rag_tool, web_search_tool)

    # Create agents
    agents = create_agents()

    # Create tasks
    tasks = create_tasks(agents, tools)

    # Create Crew
    rag_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
    )

    # Run the pipeline
    try:
        result = rag_crew.kickoff(inputs={"question": question})
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio Interface
def gradio_interface(query):
    if not query:
        return "Please enter a question."
    return run_rag_pipeline(query)

# Create Gradio App
def create_gradio_app():
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(label="Enter your question about AI, Language Models, or Self-Attention"),
        outputs=gr.Textbox(label="Response"),
        title="Agentic RAG Demo: AI Knowledge Assistant",
        description="Ask questions about AI, Language Models, Transformers, and Self-Attention mechanisms. The system uses a multi-agent approach to retrieve and verify information.",
        theme="soft",
        allow_flagging="never"
    )
    return iface

# Launch the Gradio App
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()