import json
import tiktoken
import locale

locale.setlocale(locale.LC_ALL, '')

tokenizer = tiktoken.get_encoding("cl100k_base")

# list your prepped JSONL files here
file_paths = ["/content/fine-tuning-data-jsonl/dataset-70_prepared.jsonl",
              "/content/fine-tuning-data-jsonl/dataset-15a_prepared.jsonl",
              "/content/fine-tuning-data-jsonl/dataset-15b_prepared.jsonl"]

token_costs = {
    "Ada": {"training": 0.0004, "usage": 0.0016},
    "Babbage": {"training": 0.0006, "usage": 0.0024},
    "Curie": {"training": 0.0030, "usage": 0.0120},
    "Davinci": {"training": 0.0300, "usage": 0.1200}
}

for file_path in file_paths:
    prompt_tokens = []
    completion_tokens = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)

        prompt_tokens.append(len(tokenizer.encode(data['prompt'])))
        completion_tokens.append(len(tokenizer.encode(data['completion'])))

    total_tokens = sum(prompt_tokens) + sum(completion_tokens)

    model_costs = {}
    for model, costs in token_costs.items():
        training_cost = costs["training"] * (total_tokens / 1000)
        usage_cost = costs["usage"] * (total_tokens / 1000)
        model_costs[model] = {"training_cost": training_cost, "usage_cost": usage_cost}

    total_tokens_str = locale.format_string("%d", total_tokens, grouping=True)

    prompt_tokens_str = locale.format_string("%d", sum(prompt_tokens), grouping=True)
    completion_tokens_str = locale.format_string("%d", sum(completion_tokens), grouping=True)

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