from langgraph.prebuilt import create_react_agent   
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv                       
from langchain_core.tools import tool                
from langchain_core.messages import HumanMessage    
from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool

load_dotenv(override=True)



chunks = [
    "Hajar is a Software Engineer with 5 years of experience, specializing in Python and JavaScript.",
    
    "She has worked on a variety of projects, building scalable and efficient applications across different domains.",
    
    "Hajar is recognized for her strong problem-solving skills and her ability to quickly understand and resolve complex technical challenges.",
    
    "She communicates technical concepts clearly and effectively, making her a valuable bridge between technical and non-technical teams.",

    "A collaborative team player, she actively contributes to team success and is always willing to support her colleagues.",

    "She thrives in dynamic environments and continuously seeks to improve her skills and deliver high-quality solutions."
]

embedding_model = OpenAIEmbeddings(model= "text-embedding-ada-002")

vector_store = Chroma.from_texts(
    texts=chunks, 
    embedding=embedding_model, 
    collection_name="cv_formation")

retriever=vector_store.as_retriever(kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_search",
    description="Use this tool to search information about me"
)


@tool                                                
def get_employee_info(name: str):
    """
    Get information about a given employee (name, salary, seniority)
    """
    print("get_employee_info tool invoked")
    return {"name": name, "salary": 12000, "seniority": 5}

@tool                                                
def send_email(email: str, subject: str, content: str):
    """
    Send email with subject and content
    """
    print(f"Sending email to {email}, subject {subject}, content {content}")
    return f"Email sent successfully to {email}, subject {subject}, content {content}"  

llm = ChatOpenAI(model="gpt-4o", temperature=0)

graph = create_react_agent(
    model=llm,                                       
    tools=[get_employee_info, retriever_tool, send_email],
    prompt="Answer to user query using provided tools"  
)

resp = graph.invoke(input={"messages": [HumanMessage(content="Quel est le salaire de Hajar?")]})
print(resp['messages'][-1].content)
