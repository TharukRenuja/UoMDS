from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_pointwise_to_file(data: str, filename: str = "output_points.txt"):
    """
    Save the given data to a file in a point-wise format.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    points = data.strip().split('\\n')
    formatted_points = "\\n".join([f"{idx+1}. {point.strip()}" for idx, point in enumerate(points) if point.strip()])
    formatted_text = f"--- Output Points ---\\nTimestamp: {timestamp}\\n\\n{formatted_points}\\n\\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Output successfully saved to {filename}"

# Define the save tool
save_tool = Tool(
    name="save_pointwise_to_file",
    func=save_pointwise_to_file,
    description="Save output in point-wise format to a text file."
)

# Define the DuckDuckGo search tool
duckduckgo_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=duckduckgo_search.run,
    description="Search the web for information."
)

# Define the Wikipedia search tool
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
