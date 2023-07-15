import openai
import pandas as pd
from google.colab import files
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

openai.api_key = "<API_KEY>"

# this is the file you will chunk and iterate over each chunk to fine-tuned model
filename = '<FILE_PATH>'

output_filename = f"UV-V1_output_{filename.split('/')[-1]}"

with open(filename, 'r') as f:
    article_text = f.read()

def recursive_text_split(text):
    sections = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=5,
        length_function=len,
    ).split_text(text)

    df = pd.DataFrame({'section': sections})
    return df
# I tokenize so I can get token count of chunk and ensure its under 1k
def run_api_call(df):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    generated_texts = []

    for _, row in df.iterrows():
        section = row['section']
        combined_prompt = f'{section}\n\n###\n\n'

        token_count = len(tokenizer.encode(combined_prompt))
        # I am keeping my input text "prompt" under 1k so i can save 1k for completon (Fine-tuned Model limit is 2k tokens)
        if token_count > 1000:
            print("Tokens exceed the limit of 1000. Stopping the code execution.")
            break

        completion = openai.Completion.create(
            model="<FINE_TUNED_MODEL_NAME>",
            prompt=combined_prompt
        )

        generated_text = completion['choices'][0]['text']
        generated_texts.append(generated_text)

    return generated_texts


df_sections = recursive_text_split(article_text)

generated_texts = run_api_call(df_sections)

# outputs saved to file and downloaded
with open(output_filename, 'w') as output_file:
    for generated_text in generated_texts:
        output_file.write(generated_text)
        output_file.write("\n\n")

files.download(output_filename)

print(f"Output file '{output_filename}' has been generated.")