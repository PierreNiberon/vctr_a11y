from redis import from_url, Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from transformers import T5Tokenizer, T5Model, pipeline, T5ForConditionalGeneration
from redis.commands.search.query import Query
import numpy as np
import torch
import wikipediaapi

# Connection to the redis instance
REDIS_URL = 'redis://localhost:6379'
client = from_url(REDIS_URL)

# ping redis instance
def redis_ping():
    return client.ping()


# instantiate the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained('t5-3b')
model = T5Model.from_pretrained('t5-3b')
generative_model = T5ForConditionalGeneration.from_pretrained("t5-3b")

# helper function to embbed text
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        # Pass the input to the encoder
        encoder_outputs = model.get_encoder()(input_ids)
        
    # Retrieve the last hidden state of the encoder
    hidden_state = encoder_outputs.last_hidden_state

    # Compute the mean over the sequence dimension
    return hidden_state.mean(dim=1).numpy().flatten().tolist()

# helper function to split big text in small chunks
def chunk_text(text, tokenizer, max_length):

    text = text.replace(".", " .")

    tokens = tokenizer.encode(text, truncation=False, return_tensors='pt')[0]
    chunks = []

    # Initialize the start index
    start = 0

    # Loop over the tokens and create chunks
    while start < len(tokens):
        if len(tokens[start:]) > max_length:
            end = start + max_length
            # Make sure the end index is not in the middle of a sentence
            while tokens[end] != tokenizer.encode(".", add_special_tokens=False)[0]:
                end -= 1
            end += 1  # Include the period in the current chunk
        else:
            end = len(tokens)

        # Create a chunk and add it to the list of chunks
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))

        # Update the start index
        start = end

    return chunks

# get wikipedia content from wki page name
def wiki_page_content(page_name):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(page_name)
    return page_py.text


# function to generate the document to load into redis
def load_redis(text):
    # chunkify the text before sending it to the tokenizer
    chunks = chunk_text(text, tokenizer, 512)
    try:
        # Builds the json with the content in natural language and the vectors
        for count, chunk in enumerate(chunks):
            doc_json = {"content": chunk, "vector": get_vector(chunk)}
            
            # Creates the schema in redis based on first doc
            if count == 0:
                schema = [ VectorField('$.vector', 
                            "FLAT", 
                            {   "TYPE": 'FLOAT32', 
                                "DIM": len(doc_json['vector']), 
                                "DISTANCE_METRIC": "COSINE"
                            },  as_name='vector' ),
                            TextField('$.content', as_name='content')
                        ]
                idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['doc:'])
                
                # Only create the index if it doesn't exist.
                try:
                    client.ft('idx').info()
                except:
                    client.ft('idx').create_index(schema, definition=idx_def)

            # get the number of keys already loaded if any
            r = Redis()
            total_keys = r.dbsize()

            # Loads the document into Redis.
            # Careful with the prefix in the name as Redis use that to associate a document and an index.
            doc_name = "doc:" + str(count + total_keys)
            client.json().set(doc_name, '$', doc_json)
        return "200"
    except:
        return "400"
    
    
def get_answer(idx, question):

    # Get vector for the question, need to be converted tobytes for redis
    vec = np.array(get_vector(question), dtype=np.float32).tobytes()
        
    
    # Define the search query
    q = Query('*=>[KNN 1 @vector $query_vec AS vector_score]').return_fields('content').dialect(2)    
    
    # Define query parameters
    params = {"query_vec": vec}

    # Execute the search query
    results = client.ft(idx).search(q, query_params=params)

    if len(results.docs) == 0:
        return "No relevant documents found in database. Please seek professional help."
    else:
        # Retrieve the content of the most relevant document
        document = results.docs[0]['content'].strip()
        print(document)
        # Build the prompt
        
        prompt = f"question: {question} context: {document}"
        print(prompt)
        # Then, pass the model and tokenizer to the pipeline function
        text2text_generator = pipeline("text2text-generation", model=generative_model, tokenizer=tokenizer)
        result = text2text_generator(prompt)
        return result

