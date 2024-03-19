import argparse
import random
import re
import datetime
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Pix2StructConfig, Adafactor, get_cosine_schedule_with_warmup
from model.doc2command import Doc2Command
from doceditprojectmodel.dataset import Dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="DocEdit Project Training Script")
    parser.add_argument("--model_directory", default="", help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=1000, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--seed_val", type=int, default=42, help="Random seed value")
    parser.add_argument("--sample_every", type=int, default=100, help="Sample and save the model every N steps")
    parser.add_argument("--dataset", default="", help="Path to the dataset")
    parser.add_argument("--model_string", default="google/pix2struct-docvqa-base", help="Pretrained model string")
    parser.add_argument("--max_patches", type=int, default=1024, help="Maximum number of patches")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--weight_decay", type=float, default=1e-05, help="Weight decay for optimizer")
    parser.add_argument("--n_cls", type=int, default=3, help="n_cls for masktransformer_config")
    parser.add_argument("--patch_size", type=int, default=16, help="patch_size for masktransformer_config")
    parser.add_argument("--d_encoder", type=int, default=768, help="d_encoder for masktransformer_config")
    parser.add_argument("--n_layers", type=int, default=12, help="n_layers for masktransformer_config")
    parser.add_argument("--n_heads", type=int, default=12, help="n_heads for masktransformer_config")
    parser.add_argument("--d_model", type=int, default=768, help="d_model for masktransformer_config")
    parser.add_argument("--d_ff", type=int, default=256, help="d_ff for masktransformer_config")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="drop_path_rate for masktransformer_config")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout for masktransformer_config")
    return parser.parse_args()

def initialize_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def save_config(config, model_directory, formatted_datetime):
    config.save_pretrained(os.path.join(model_directory, f"config_{formatted_datetime}"))

def load_config(model_directory, config_class):
    return config_class.from_pretrained(model_directory)

def train(epochs, train_loader, val_loader, model, optimizer, scheduler, device):
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        print(f"Epoch {epoch}")
        total_train_loss = 0
        total_decoder_loss = 0
        total_mask_loss = 0


        model.train()

        train_batch_tqdm = tqdm(
            train_loader, desc="Epoch {}/{}".format(epoch + 1, epochs), leave=False
        )
        for batch_idx, batch in enumerate(train_batch_tqdm):

            batch = {k: batch[k].to(device) if not isinstance(batch[k], tuple) else batch[k] for k in batch}
            outputs = model(
                **batch
            )

            loss = outputs["loss"]

            loss.backward()

            optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            total_decoder_loss += outputs["decoder_loss"].item()
            total_mask_loss += outputs["mask_loss"].item()
            train_batch_tqdm.set_postfix(loss=loss.item())

        print(f"Total Train Loss {total_train_loss}")
        print(f"Total decoder loss {total_decoder_loss}")
        print(f"Total mask loss {total_mask_loss}")

        total_val_loss = 0
        model.eval()
        total_decoder_loss = 0
        val_batch_tqdm = tqdm(val_loader, desc="Validation", leave=False)

        for batch_idx, batch in enumerate(val_batch_tqdm):
            with torch.no_grad():
                batch = {k: batch[k].to(device) if not isinstance(batch[k], tuple) else batch[k] for k in batch}

                outputs = model(
                **batch
                )

                loss = outputs["loss"]
                total_val_loss += loss.item()
                total_decoder_loss += outputs["decoder_loss"].item()
                total_mask_loss += outputs["mask_loss"].item()
                val_batch_tqdm.set_postfix(loss=loss.item())

        

        print(f"Total Val Loss {total_val_loss}")
        print(f"Total Val decoder loss {total_decoder_loss}")
        print(f"Total Val mask loss {total_mask_loss}")

        try:
            if (epoch+1)%10==0:
                torch.save(model.state_dict(), os.path.join(model_directory, f"model_p2s_{formatted_datetime}__{epoch+1}.pt"))
        except:
            print(f"save failed at epoch {epoch}")

def main():
    args = parse_arguments()

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H-%M-%S")
    file_name = f"model_p2s_{formatted_datetime}.pt"
    model_file = os.path.join(args.model_directory, file_name)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    initialize_seed(args.seed_val)

    num_warmup_steps = args.num_warmup_steps
    batch_size = args.batch_size
    epochs = args.epochs
    max_steps = int(12450 / batch_size * epochs)

    learning_rate = args.learning_rate
    epsilon = 1e-8
    sample_every = args.sample_every

    train_path = os.path.join(args.dataset, "train_merge.csv")
    val_path = os.path.join(args.dataset, "val_merge.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    processor = AutoProcessor.from_pretrained(args.model_string)
    p2s_config = Pix2StructConfig.from_pretrained(args.model_string)

    def collator(batch):
        new_batch = {"flattened_patches": [], "attention_mask": [], "shape": [], "ground_truth_mask":[]}
        outputs = [item["output"] for item in batch]

        text_outputs = processor.tokenizer(
            text=outputs,
            add_special_tokens=False,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


        new_batch["labels"] = text_outputs.input_ids


        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
            new_batch["ground_truth_mask"].append(item["ground_truth_mask"])
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["shape"] = batch[0]["shape"]
        new_batch["ground_truth_mask"] = torch.stack(new_batch["ground_truth_mask"])

        return new_batch


    masktransformer_config = {
        "n_cls": args.n_cls,
        "patch_size": args.patch_size,
        "d_encoder": args.d_encoder,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "drop_path_rate": args.drop_path_rate,
        "dropout": args.dropout
    }

    config_dict = {
        "masktransformer_config": masktransformer_config,
        "model_string": args.model_string,
        "p2s": p2s_config
    }

    model = Doc2Command(config_dict)

    processor.image_processor.is_vqa = True
    tokenizer = processor.tokenizer

    train_dataset = Dataset(train_df, processor, args.dataset)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=8)

    val_dataset = Dataset(val_df, processor, args.dataset)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collator, num_workers=8)

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=False,
        relative_step=False,
        lr=learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps
    )

    save_config(p2s_config, args.model_directory, formatted_datetime)  # Save config before training
    train(epochs, train_loader, val_loader, model, optimizer, scheduler, device)

if __name__ == "__main__":
    main()
