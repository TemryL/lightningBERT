import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging


def chunk_doc(examples, indices, tokenizer, chunk_length):
    chunks = []
    doc_idxs = []
    chunk_idxs = []
    attention_masks = []
    
    pad_token_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    for example, doc_idx in zip(examples['text'], indices):
        tokens = tokenizer.encode(example, add_special_tokens=False)
        
        for idx, i in enumerate(range(0, len(tokens), chunk_length - 2)):
            chunk = tokens[i:i + chunk_length - 2]
            
            # Add CLS and SEP tokens
            padded_chunk = [cls_id] + chunk + [sep_id]
            
            # Pad if necessary
            padding_length = chunk_length - len(padded_chunk)
            if padding_length > 0:
                padded_chunk += [pad_token_id] * padding_length
            attention_mask = [1] * (chunk_length - padding_length) + [0] * padding_length
            
            assert len(padded_chunk) == chunk_length
            assert len(attention_mask) == chunk_length
            
            chunks.append(padded_chunk)
            doc_idxs.append(doc_idx)
            chunk_idxs.append(idx)
            attention_masks.append(attention_mask)
    
    return {
        'token_ids': chunks,
        'attention_mask': attention_masks,
        'doc_idx': doc_idxs,
        'chunk_idx': chunk_idxs
    }


def main():
    CHUNK_LENGTH = 128
    TOKENIZER_NAME = 'bert-base-uncased'

    # Load the tokenizer
    logging.set_verbosity(40)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load the dataset
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    
    # Create the dataset using map function
    tokenized_dataset = dataset.map(
        lambda examples, indices: chunk_doc(examples, indices, tokenizer, CHUNK_LENGTH),
        remove_columns=dataset.column_names,
        batched=True,
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Creating tokenized dataset"
    )

    # Push the dataset to the Hugging Face Hub
    dataset_name = f"tokenized_wikipedia_20220301.en_train_{CHUNK_LENGTH}"
    tokenized_dataset.push_to_hub(dataset_name, token=os.environ.get("HF_TOKEN"))
    print(f"Dataset pushed to Hugging Face Hub: {dataset_name}")


if __name__ == '__main__':
    main()