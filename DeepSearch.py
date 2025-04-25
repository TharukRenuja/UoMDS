from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    title: str
    summary: str
    sources: list[str]
    tools: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main(query):
    save_flag = False
    if "[save to a file]" in query.lower():
        save_flag = True
        query = query.lower().replace("[save to a file]", "").strip()
    try:
        raw_response = agent_executor.invoke({"query": query})
        print("Raw response:", raw_response)  # Debug print to inspect structure
        # Adjust parsing based on raw_response structure
        output = raw_response.get("output")
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
            output_text = output[0].get("text", "")
        elif isinstance(output, str):
            output_text = output
        else:
            output_text = ""
        if output_text.startswith("```json"):
            output_text = output_text[len("```json"):].strip()
        if output_text.endswith("```"):
            output_text = output_text[:-3].strip()
        # Unescape newline characters
        output_text = output_text.replace("\\n", "\n")
        structured_response = parser.parse(output_text)
        print("Research Response:")
        print(f"Title: {structured_response.title}")
        print("Summary:")
        print(structured_response.summary)
        print("Sources:", structured_response.sources)
        print("Tools used:", structured_response.tools)
        if save_flag:
            save_result = save_tool.func(structured_response.summary)
            print(save_result)
        return structured_response
    except Exception as e:
        import traceback
        print("Sorry, an error occurred while processing your request.")
        print("Error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
