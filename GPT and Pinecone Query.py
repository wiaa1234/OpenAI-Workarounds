import openai
import pinecone

# OpenAI params
embed_model = "text-embedding-ada-002"
openai.api_key = "<API_KEY>"

# Pinecone params
PINECONE_API_KEY = '<API_KEY>'
PINECONE_API_ENV = '<ENV_KEY>'
INDEX_NAME = '<INDEX>'
PINECONE_NAMESPACE = '<NAMESPACE>'
METADATA_1 = { "$and": [{ "genre": "comedy" }, { "genre": "drama" }] }
METADATA_2 = {"year": 2019}

# User-defined messages
system_msg = "You are a Marketing Copywriter."
user_message = "Write a short marketing copy piece."

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index = pinecone.Index(INDEX_NAME)

embed_query = openai.Embedding.create(
    input=[user_message],
    engine=embed_model
)

query_embeds = embed_query['data'][0]['embedding']

filter_criteria = {}
filter_criteria.update(METADATA_1)
filter_criteria.update(METADATA_2)

response = index.query(
    query_embeds,
    top_k=5,
    include_metadata=True,
    namespace=PINECONE_NAMESPACE,
    filter=filter_criteria
)

matches = response['matches']

contexts = [item['metadata']['vosst_value'] for item in response['matches']]

augmented_query = " --- ".join(contexts) + " --- " + user_message

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": augmented_query}
]

chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=messages
)

assistant_message = chat['choices'][0]['message']['content']
messages.append({"role": "assistant", "content": assistant_message})

print(assistant_message, matches)

# Save assistant message to output file
output_filename = 'pinecone-gpt-query.txt'
with open(output_filename, 'a') as output_file:
    output_file.write(assistant_message)
    output_file.write("\n\n")