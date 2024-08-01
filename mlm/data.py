import random
import torch
import lightning as L
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import BertTokenizer


class WikipediaMLMDataset(Dataset):
    def __init__(self, mlm_probability=0.15):
        self.data = load_dataset("TemryL/tokenized_wikipedia_20220301.en_train_512", split='train')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 512
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]['token_ids']
        attention_mask = self.data[idx]['attention_mask']
        
        # Ensure the chunk and attention mask is exactly max_length
        assert len(chunk) == self.max_length
        assert len(attention_mask) == self.max_length
        
        # Create input_ids and labels for MLM
        input_ids = torch.tensor(chunk)
        attention_mask = torch.tensor(attention_mask)
        labels = input_ids.clone()
        
        # Mask random tokens (only where attention_mask is 1)
        for i in range(len(input_ids)):
            if attention_mask[i] == 1 and random.random() < self.mlm_probability:
                if random.random() < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class WikipediaMLMDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4, 
                 train_val_split: float = 0.95,
                 mlm_probability: float = 0.15,
                 pin_memory: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.mlm_probability = mlm_probability
        self.pin_memory = pin_memory
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = WikipediaMLMDataset(mlm_probability=self.mlm_probability)
        
        if stage == 'fit' or stage is None:
            train_size = int(len(self.dataset) * self.train_val_split)
            val_size = len(self.dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size]
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)