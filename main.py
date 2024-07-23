#region ##### Load packages #####
import semantic_kernel as sk
import asyncio
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
 
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
 
from semantic_kernel.functions import KernelArguments
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
 
from azure.core.credentials import AzureKeyCredential
 
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType, VectorizableTextQuery
 
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import dotenv_values
#endregion
import os
#region ##### Initialise app #####
app = Flask(__name__)
cors = CORS(app, origins='*')
#endregion
 
#region ###### Load environment variables #####
# print("SECRETS")
secrets=dotenv_values(".env")
 
 
print("This Happened")
 
#endregion
 
 
# OPENAI_ENDPOINT = secrets["OPENAI_ENDPOINT"]
# OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
# OPENAI_CHAT_DEPLOYMENT_NAME = secrets["OPENAI_CHAT_DEPLOYMENT_NAME"]
# OPENAI_EMBEDDING_DEPLOYMENT_NAME = secrets['OPENAI_EMBEDDING_DEPLOYMENT_NAME']
# AZURE_SUBSCRIPTION_KEY = secrets["AZURE_SUBSCRIPTION_KEY"] 
# AZURE_REGION = secrets["AZURE_REGION"] 

OPENAI_ENDPOINT = os.environ["OPENAI_ENDPOINT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_CHAT_DEPLOYMENT_NAME = os.environ["OPENAI_CHAT_DEPLOYMENT_NAME"]
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_DEPLOYMENT_NAME']
# AZURE_SUBSCRIPTION_KEY = os.environ["AZURE_SUBSCRIPTION_KEY"] 
# AZURE_REGION = os.environ["AZURE_REGION"] 
 
#endregion
 
#region ##### Create Kernel and Add Services #####
kernel = sk.Kernel()
 
 
chat_service_id="azure_gpt35_chat_completion"
#Add services to kernel
kernel.add_service(
        service=AzureChatCompletion(
        service_id=chat_service_id,
        deployment_name=OPENAI_CHAT_DEPLOYMENT_NAME,
        endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        )
    )
 
#endregion
 
#region ##### Chat Function Creation #####
# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(chat_service_id)
req_settings.max_tokens = 4000
req_settings.temperature = 0.7
req_settings.top_p = 0.5
 
prompt = """
    You are a chatbot that can have a conversations about any topic related to the Provided Context and History, returning all relevant details of any questions asked.
    Review the History and Provided Context for information that can help you in forming a response.
    Give explicit answers to the questions asked or say you don't know and require more information (e.g.'I don't know. Please provide more details') if it does not have an answer or you are not sure.
         
    Here is a description of the output you should always provide:
   
    Response: "The Chatbot System response with relevant numbered citations at the end of each paragraph",
    Citations: "The list of numbered citations that are in the response followed by their individual Document_Name"}    
   
    Here is a sample response provided between the "___" areas below. Never deviate from this structure.
    _______________________________________________
    Response:
    Sample Text [1] \n
    Sample Text [2] \n
    Sample Text [1][3][4]
     
    Citations:
    [1] Document_Name
    [2] Document_Name
    [3] Document_Name
    [4] Document_Name
    _______________________________________________
   
    Always include associated citations numbers for each paragraph. A paragraph should always have a minimum of one citation.
    You should reuse the unique numbered citation IDs in your response where relevant.
    The "Citations" section should only have IDs and ther related Document_Names. There cannot be a citation without an ID number.
    The final set of Citations IDs are numeric and always start from 1 and increase in increments of 1 only.
   
    If content from a document is not included in the final system response then do not include it in the citations.
   
    Provided Context: {{$db_record}}
 
    History: {{$history}}
   
    User: {{$query_term}}
   
    Chatbot:
    """
       
 
chat_prompt_hist_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat_with_history",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="db_record", description="The database record", is_required=True),
            InputVariable(name="query_term", description="The user input", is_required=True),
            InputVariable(name="history", description="The chat histroy", is_required=True),
        ],
        execution_settings=req_settings,
    )
 
chat_with_history_function = kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt=prompt,
        prompt_template_config=chat_prompt_hist_template_config
    )
 
#endregion
 
#region Generate Multiple Vector Queries Function Creation
search_prompt = """
    You are a chatbot that can generate multiple rephrased versions of the sentence you are supplied to optimize vector queries in document search.
    Always return your ouput in the format: ['Query 1', 'Query 2', .... etc]
    You will rephrase the sentence provided in the query term:
   
    User: {{$query_term}}
    Chatbot:"""
 
 
search_prompt_query_config = PromptTemplateConfig(
    template=search_prompt,
    name="generate_vector_queries",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=req_settings,
)
 
generate_vector_queries_function = kernel.add_function(
    plugin_name="VQCreation",
    function_name="Queries",
    prompt=search_prompt,
    prompt_template_config=search_prompt_query_config
)
#endregion
#region ##### Create AI Search function #####
 #Azure AI Search Credentials
# AZURE_SEARCH_ENDPOINT = secrets["AZURE_SEARCH_ENDPOINT"]
# AZURE_SEARCH_API_KEY = secrets["AZURE_SEARCH_API_KEY"]
# AZURE_INDEX = secrets["AZURE_INDEX"]

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_API_KEY = os.environ["AZURE_SEARCH_API_KEY"]
AZURE_INDEX = os.environ["AZURE_INDEX"]
 
 
AZURE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_SEARCH_API_KEY)
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_INDEX, AZURE_SEARCH_CREDENTIAL)
   
def Search(query_term,
           vector_qs, search_client):
 
   
    #AI Search with Semantic Ranker
    search_result = search_client.search(
        search_text= query_term,
        vector_queries=vector_qs,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=10,
        search_fields=["chunk"]
    )
   
    #Extract data from search result iterator object
    data = [{"document_id": result["id"],
            "document_name": result["title"],
             "chunk_id":result["chunk_id"],
             "chunk":result["chunk"],
             "search_score":result["@search.score"],
             "semantic_rank":result['@search.reranker_score'],
             "caption": result["@search.captions"],
             "chunk_links": result['links'],
             "document_links" : result['FullDocLinks']}
            for result in search_result]
   
    return(data)
#endregion
 
#region Memory
 
memory_collection_id = "User Session Chat History"
 
embedding_gen = AzureTextEmbedding(
      service_id="ada",
      deployment_name=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
      endpoint=OPENAI_ENDPOINT,
      api_key=OPENAI_API_KEY,
    )
kernel.add_service(embedding_gen)
 
 
#Function to update memory
async def populate_memory(memory: SemanticTextMemory, text_: str, memory_id: str, collection_id=memory_collection_id) -> None:
    # Add some documents to the semantic memory
    await memory.save_information(collection=collection_id, id=memory_id, text=text_)
 
# Create the memory store
memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)
 
# Create the initial memory for the collection
async def initialise_history(memory_store, collection_id):
       await memory_store.save_information(collection=collection_id, id="Core Memory", text="System: You are a helpful chatbot that can politely and professionally assist with searching for information in documents.")
 
asyncio.run(initialise_history(memory, memory_collection_id))
 
saved_memories = []
 
async def RemoveMemories(memory_store: SemanticTextMemory, saved_memories:list, collection_id=memory_collection_id):
    await memory_store._storage.remove_batch(collection_name=collection_id, keys = saved_memories)
 
session_cost = 0
#region ##### Main API call #####
@cross_origin()
@app.route('/api/chat', methods=['POST'])
async def chat():
    global saved_memories
    #Get frontend input
    query_term = request.json['message']
   
    #region reset memory
    if query_term == "Clear Chat":
        #delete all saved memories
       
        await RemoveMemories(memory_store=memory, saved_memories=saved_memories)
       
        #reset saved memories to empty
        # global saved_memories
        saved_memories = []
       
        return
    #endregion
   
    #region Perform AI Search using input
 
    #Generate multiple user queries to improve embedding search
    generated_queries =  await kernel.invoke(
        generate_vector_queries_function, KernelArguments(query_term=query_term)
    )
    vector_queries = [VectorizableTextQuery(text=query,
                                            k_nearest_neighbors=50,
                                            fields="vector",
                                            exhaustive=True)
                      for query
                      in generated_queries]
   
    #AI Search with Semantic Ranker
    search_result = search_client.search(
        search_text= query_term,
        vector_queries=vector_queries,
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=10,
        search_fields=["chunk"]
    )
    #endregion
   
    #region Formt Serch Output
    #Extract data from search result iterator object
    data = [{"document_id": result["id"],
            "document_name": result["title"],
             "chunk_id":result["chunk_id"],
             "chunk":result["chunk"],
             "search_score":result["@search.score"],
             "semantic_rank":result['@search.reranker_score'],
             "caption": result["@search.captions"],
             "chunk_links": result['links'],
             "document_links" : result['FullDocLinks'],
             "doc_header" : result["entities"]}
            for result in search_result]
 
   
    db_metadata = {}
    [db_metadata.update({record["document_name"] : record["doc_header"][0].split('"Source URL": "')[1].split('"')[0] for record in data}) ]
 
       
    #Create a single string of db_content
    db_contents = "".join(['[[This Document_Name is ' + record["document_name"] + ' and should be provided as the only citation for this information:' +  record["chunk"] + ']]' for record in data])
    #endregion
   
    #region URL Extraction
    #Pull the URLs directly referenced in the chunk
    db_chunk_urls = {} #initialise empty dictionary
    [
    db_chunk_urls[record["document_name"]].update(record["chunk_links"])
    if record["document_name"] in db_chunk_urls
    else db_chunk_urls.update({record["document_name"]: set(record["chunk_links"])})
    for record in data
    ]
    db_chunk_urls = {doc_name: list(links) for doc_name, links in db_chunk_urls.items()}
 
    #Get a unique list of document links that may be related to the chunks used in the answer context from their paretn documents
    db_doc_urls = {}
    [
    db_doc_urls[record["document_name"]].update(record["document_links"])
    if record["document_name"] in db_doc_urls
    else db_doc_urls.update({record["document_name"]: set(record["document_links"])})
    for record in data
    ]
    db_doc_urls = {doc_name: list(links) for doc_name, links in db_doc_urls.items()}
    #endregion
   
    #region Perform AzureOpenAI API call with relevant prompt requirements
    #Pull the top N relevant memories from the semantic memory store for the prompt
    memories = await memory.search(memory_collection_id, query_term, limit=3, min_relevance_score=0.7)
    chat_history = "".join([str(memory.text) for memory in memories])
   
    #Provide context to Chat Model to answer query
    completions_result =  await kernel.invoke(
        chat_with_history_function, KernelArguments(query_term=query_term, db_record=db_contents, history=chat_history)
    )
   
    citations_list = {}
    [citations_list.update({key: value}) for key,value in db_metadata.items()]
    split_results = str(completions_result).split("Citations:")
    json_results = {"message": split_results[0],
                    "citations": citations_list}
   
    #Update the memory with the most recent interaction
    await populate_memory(memory=memory, memory_id=query_term, collection_id=memory_collection_id,
                    text_ = "".join([
                        "User: ",
                        query_term,
                        "\n\n"
                        "System: ",
                        str(completions_result)])  
                    )
   
    # update global saved_memories
    saved_memories.append(query_term)
    #endregion
 
    #region Costing Estimate
    #The cost of the input prompt (including db_content + History)
    # prompt_tokens = completions_result.metadata["metadata"][0]["usage"].prompt_tokens
    # prompt_cost_estimate = int(prompt_tokens)*0.002/1000,
   
    # #The response cost
    # completion_tokens = completions_result.metadata["metadata"][0]["usage"].completion_tokens
    # completion_cost_estimate = int(completion_tokens)*0.002/1000
   
    # global session_cost
    # session_cost += prompt_cost_estimate + completion_cost_estimate
    #endregion
 
 
    #Format the output to be sent to frontend
    response = json_results
   
    return(response)

# @app.route('/api/config')
# def get_config():
#     return jsonify({
#         'azure_subscription_key': AZURE_SUBSCRIPTION_KEY,
#         'azure_region': AZURE_REGION
#     })

 
#endregion
 
if __name__ == '__main__':
    app.run(host="127.0.0.1", port="5000", debug=True)