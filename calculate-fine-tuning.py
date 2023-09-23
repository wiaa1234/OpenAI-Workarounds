import json
import tiktoken
import locale

# Set the locale to use commas for thousands separator
locale.setlocale(locale.LC_ALL, '')

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Specify the JSONL file paths that you got from OpenAI CLI Data-Prep Tool
file_paths = ["<file_path_here>",
              "<file_path_here>",
              "<file_path_here>"]

# Define the token costs for each model
token_costs = {
    "Ada": {"training": 0.0004, "usage": 0.0016},
    "Babbage": {"training": 0.0006, "usage": 0.0024},
    "Curie": {"training": 0.0030, "usage": 0.0120},
    "Davinci": {"training": 0.0300, "usage": 0.1200}
}

# Loop over the files
for file_path in file_paths:
    # Initialize lists to store token counts
    prompt_tokens = []
    completion_tokens = []

    # Open the file and read in lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Loop over the lines
    for line in lines:
        # Parse line as JSON
        data = json.loads(line)

        # Tokenize 'prompt' and 'completion' fields and count the tokens
        prompt_tokens.append(len(tokenizer.encode(data['prompt'])))
        completion_tokens.append(len(tokenizer.encode(data['completion'])))

    # Calculate the total number of tokens for the file
    total_tokens = sum(prompt_tokens) + sum(completion_tokens)

    # Calculate the cost for each model based on the token counts
    model_costs = {}
    for model, costs in token_costs.items():
        training_cost = costs["training"] * (total_tokens / 1000)
        usage_cost = costs["usage"] * (total_tokens / 1000)
        model_costs[model] = {"training_cost": training_cost, "usage_cost": usage_cost}

    # Format the total token count with commas
    total_tokens_str = locale.format_string("%d", total_tokens, grouping=True)

    # Format the prompt and completion token counts with commas
    prompt_tokens_str = locale.format_string("%d", sum(prompt_tokens), grouping=True)
    completion_tokens_str = locale.format_string("%d", sum(completion_tokens), grouping=True)

    # Print the token counts and costs for each model for the file
    print(f"File: {file_path}")
    print(f"Total Tokens: {total_tokens_str}")
    print(f"Prompt Tokens: {prompt_tokens_str}")
    print(f"Completion Tokens: {completion_tokens_str}")
    print()
    for model, costs in model_costs.items():
        training_cost_str = locale.currency(costs["training_cost"], grouping=True)
        usage_cost_str = locale.currency(costs["usage_cost"], grouping=True)
        print(f"Model: {model}")
        print(f"Training cost: {training_cost_str}")
        print(f"Usage cost: {usage_cost_str}")
        print()
