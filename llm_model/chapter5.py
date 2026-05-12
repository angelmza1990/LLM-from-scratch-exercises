import argparse
import os

import requests
import torch
import tiktoken

from chapter4 import GPTModel, generate_text_simple
from dataloader import create_dataloader_v1

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer, prompt):
    train_losses, val_losses = [], []
    tokens_seen, global_step = 0, 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} | Step {global_step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        generate_sample(model, tokenizer, prompt, device)

    return train_losses, val_losses


def generate_sample(model, tokenizer, prompt, device, max_new_tokens=25, context_size=None):
    model.eval()
    if context_size is None:
        context_size = model.pos_emb.weight.shape[0]
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        output_ids = generate_text_simple(model, idx, max_new_tokens, context_size)
    decoded = token_ids_to_text(output_ids, tokenizer)
    print("\n=== Generated text ===")
    print(decoded)
    print("======================\n")
    model.train()


def download_verdict_text(force_download=False):
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    if force_download or not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    if args.train:
        text_data = download_verdict_text()
        split_idx = int(0.9 * len(text_data))
        train_loader = create_dataloader_v1(
            text_data[:split_idx],
            batch_size=args.batch_size,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        val_loader = create_dataloader_v1(
            text_data[split_idx:],
            batch_size=args.batch_size,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            eval_freq=args.eval_freq,
            eval_iter=args.eval_iter,
            tokenizer=tokenizer,
            prompt=args.prompt,
        )

    print("\nRunning generation with random weights...")
    generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a GPT model from scratch without OpenAI pretrained weights.")
    parser.add_argument("--prompt", default="Every effort moves you", help="Prompt text to generate from.")
    parser.add_argument("--train", action="store_true", help="Train the model on the sample dataset before generation.")
    parser.add_argument("--num-epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for training.")
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluate every n steps during training.")
    parser.add_argument("--eval-iter", type=int, default=1, help="Number of batches to use for evaluation.")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Number of tokens to generate.")
    parser.add_argument("--force-download", action="store_true", help="Force download of the sample text dataset.")
    args = parser.parse_args()

    main(args)
