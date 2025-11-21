from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from BookDataset import BookDataset
import torch, re, math, requests


def main():
  url = "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"
  text = requests.get(url).text
  text = re.sub(r'(\n\s*)+', '\n', text)
  text = re.sub(r'\s+', ' ', text).strip()
  start = text.find("*** START OF")
  end = text.find("*** END OF")
  if start != -1 and end != -1:
    text = text[start:end]

  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  tokens = tokenizer.encode(text)
  print(f"Total tokens: {len(tokens):,}")

  hyperparameter_dict = {
    "block_size": [64,128,256],
    "lr": [1e-4, 5e-5, 1e-5],
    "epochs": [2, 5, 7],
    "batch_size": [1, 2, 4]
  }

  results = []

  for block_size in hyperparameter_dict["block_size"]:
    examples = []
    for i in range(0, len(tokens) - block_size - 1, block_size // 2):
      input_seq = tokens[i : i + block_size]
      label_seq = tokens[i + 1 : i + block_size + 1]
      examples.append((input_seq, label_seq))

    print("Total training sequences:", len(examples))

    train_data, val_data = train_test_split(examples, test_size=0.1, random_state=42)
    train_dataset = BookDataset(train_data)
    val_dataset = BookDataset(val_data)

    for lr in hyperparameter_dict["lr"]:
      for epochs in hyperparameter_dict["epochs"]:
        for batch_size in hyperparameter_dict["batch_size"]:
          model = AutoModelForCausalLM.from_pretrained("gpt2")

          training_args = TrainingArguments(
            output_dir="data/results",
            overwrite_output_dir=True,
            eval_strategy="epoch",
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_strategy="epoch",
            logging_steps=50,
            fp16=True,
          )

          trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
          )

          trainer.train()

          eval_results = trainer.evaluate()
          perplexity = math.exp(eval_results["eval_loss"])
          
          results.append({
            "block_size": block_size,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "perplexity": perplexity
          })

  

if __name__ == "__main__":
  main()