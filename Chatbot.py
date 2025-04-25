from logging import exception
import chromadb

from PyPDF2 import PdfReader
import google.generativeai as genAI


genAI.configure(api_key='AIzaSyBvXeIDmeMxTqqXGKN7n_9QAjiXIkQerDQ')
embedding_model_name = 'models/embedding-001'

generativeModelName ='gemini-1.5-flash'
generativeModel = genAI.GenerativeModel(generativeModelName)
client = chromadb.Client()


def break_to_chunks(perChunkSize,overlapSize,Full_text):
    start = 0
    chunks = []
    while start < len(Full_text):
        end = min(perChunkSize + start,len(Full_text))
        chunk = Full_text[start:end]
        start += perChunkSize - overlapSize
        chunks.append(chunk)
    return chunks

def get_text_in_file(path):
    text = ''
    pages = []
    chunks_with_page = [] # Changed variable name
    try:
        with open(path,'rb') as file:
            reader = PdfReader(file)
            for pageNum in range(len(reader.pages)):
                page = reader.pages[pageNum]
                page_text = page.extract_text()
                page_chunks = break_to_chunks(512, 150, page_text)
                chunks_with_page.append([(chunk, pageNum) for chunk in page_chunks]) # Store chunk with page number
                pages.append(page_text)
                text += page_text + '\n'
    except FileNotFoundError:
        print("Couldn't find the file")
        return None
    return [text, pages, chunks_with_page] # Return the list of (chunk, page_num)

full_text_pages = get_text_in_file('Data/grade-11-history-text-book.pdf')
full_text = full_text_pages[0]
full_pages = full_text_pages[1]


if full_text:
    print('Text Extraction Complete, text length:',len(full_text))
else:
    print('Extraction Failed')

chunks_with_page_nested = get_text_in_file('Data/grade-11-history-text-book.pdf')[2] # Get the new structure
print('Broken Down Chunks (nested with page):',len(chunks_with_page_nested))

# Flatten the list of lists of (chunk, page_num) tuples
chunks_with_page = [item for sublist in chunks_with_page_nested for item in sublist]
chunks = [item[0] for item in chunks_with_page] # Separate chunks for embedding
page_numbers_for_chunks = [item[1] for item in chunks_with_page] # Keep track of page numbers

print('Broken Down Chunks (flat):', len(chunks))


def start_embedding(gen_chunks,batchSize):
    embeddings = []
    for i in range(0,len(gen_chunks),batchSize):
        batch = gen_chunks[i:i+batchSize]
        response =  genAI.embed_content(model=embedding_model_name, content=batch)

        if response and response['embedding']:
            if isinstance(response['embedding'],list) and isinstance(response['embedding'][0],list):
                embeddings.extend(response['embedding'])
            elif isinstance(response['embedding'],list):
                embeddings.append(response['embedding'])
            elif response and response.get('results'):
                for result in response.get('results', []):
                    if result.get('embedding') and result['embedding'].get('values'):
                        embeddings.append(result['embedding']['values'])
            else:
                print('no values for batch:',i)
        else:
            print('embedding failed at response')

    return embeddings

embedding = start_embedding(chunks,32)
print('amount of embeddings: ',len(embedding),'\nsample:',embedding[10])


Collection_Name = 'history_grade_11_with_page' # Changed collection name
collection = client.get_or_create_collection(name=Collection_Name)

metadata = [{"chunk_id": i, "page_number": page_numbers_for_chunks[i]} for i in range(len(chunks))] # Include page number in metadata

collection.add(
    embeddings = embedding,
    documents= chunks,
    metadatas = metadata,
    ids= [f'chunk_{i}' for i in range(len(chunks))]
)

print(f"Successfully added {collection.count()} items to the ChromaDB collection '{Collection_Name}'.")

def retrieve_relevant_chunks(query,Collection,embedding_model,top_n=5):
    relevant_chunks_with_page = [] # To store (chunk, page_number)
    try:
        response = genAI.embed_content(model=embedding_model,content=query)
        if response and response['embedding']:
            query_embedding = [response['embedding']]
            results = Collection.query(
                query_embeddings=query_embedding,
                n_results=top_n,
                include=['metadatas', 'documents'] # Include metadata to get page number
            )
            relevant_documents = results['documents'][0]
            relevant_metadatas = results['metadatas'][0]

            for doc, meta in zip(relevant_documents, relevant_metadatas):
                relevant_chunks_with_page.append((doc, meta.get('page_number')))

            return relevant_chunks_with_page
        else:
            print('Error Embedding Query')
    except FileNotFoundError:
        print('No Response')


def generate_ai_answer(query,relevant_chunks_with_page,generation_model): # Changed parameter name

    context_with_page = []
    page_numbers = set()
    for chunk, page_num in relevant_chunks_with_page:
        context_with_page.append(chunk)
        page_numbers.add(page_num)

    context = "\n".join(context_with_page)
    prompt = f"""Answer the question below based on the provided context.

    Context:
    {context}

    page Numbers:
    {sorted(list(page_numbers))}

    Question:
    {query}

     give the output in this format
     Question Index: (question number)
     Question: (question)
     Answer: (answer)
     Page Number: (page number)
     context: (context)

"""

    try:
        response = generation_model.generate_content(
            contents=[prompt],
            generation_config=genAI.types.GenerationConfig(
                max_output_tokens=200
            )
        )
        if response and response.text:
            return response.text
        else:
            return 'No Answer'
    except exception:
        print("Error in generation")




# for question in questionsTxt:
#     chunkGen = retrieve_relevant_chunks(question, collection, embedding_model_name)
#     print(generate_ai_answer(question, chunkGen, generativeModel),'\n\n')
# else:
# still_running = True
# while still_running:
#     query = input('What is your Question?\n')
#     if query == 'n':
#         break
#
#     r_chunks_with_page = []
#     chunk_page_info = retrieve_relevant_chunks(query,collection,embedding_model_name)
#
#     count = 0
#     if chunk_page_info:
#         print('relevant chunks found:', len(chunk_page_info))
#     else:
#         print('Error Finding Relevant Chunks!')
#
#     print(generate_ai_answer(query, chunk_page_info, generativeModel))