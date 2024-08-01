# Code taken and modified from: https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/text-transformers.html

import datasets
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        task_name: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pin_memory: bool = True, 
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("nyu-mll/glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        self.test_splits = [x for x in self.dataset.keys() if "test" in x]

    def prepare_data(self):
        datasets.load_dataset("nyu-mll/glue", self.task_name)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers
            )
        elif len(self.eval_splits) > 1:
            return [DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers
                ) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.test_splits) == 1:
            return DataLoader(
                self.dataset["test"],
                batch_size=self.eval_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers
            )
        elif len(self.test_splits) > 1:
            return [DataLoader(
                self.dataset[x],
                batch_size=self.eval_batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers
            ) for x in self.test_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features