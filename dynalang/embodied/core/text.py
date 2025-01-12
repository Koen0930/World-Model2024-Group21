import datasets
from transformers import T5Tokenizer
import numpy as np

# 例: data_obj という Dataset を直接受け取るパラメータを追加
class BatchedTextDataset:

    def __init__(self, name=None, batch_size=16, length=128, dataset_space=None,
                 debug=False):

        if name == "roneneldan/TinyStories":
            # 従来のロジック
            if "_" in name:
                dataset = datasets.load_dataset(*name.split("_"))["train"]
            else:
                dataset = datasets.load_dataset(name)["train"]
        else:
            # 既に用意された Dataset をそのまま使う
            dataset = datasets.load_dataset("text", data_files=name)["train"]
        print(f"sample: {dataset[0]}")
        # debugオプションがあればサブセット化
        if debug:
            dataset = dataset.select(range(10000))
        dataset = dataset.shuffle(seed=42)

        tok = T5Tokenizer.from_pretrained("t5-small") 
        def tokenize(batch):
            return tok(batch["text"])

        # トークン化
        dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=4,
            remove_columns=dataset.column_names,  # "text"を削除
        )

        # 以下は元の chunk_text とか shard の処理を同じように実装
        def chunk_text(examples):
            keys = ["input_ids"]
            concat_examples = {k: sum(examples[k], []) for k in keys}
            total_length = len(concat_examples["input_ids"])
            total_length = (total_length // length) * length
            result = {
                k: [t[i : i + length] for i in range(0, total_length, length)]
                for k, t in concat_examples.items()
            }
            # result["labels"] = result["input_ids"].copy()
            result["token"] = np.array(result["input_ids"])
            return result

        dataset = dataset.map(
            chunk_text,
            batched=True,
            remove_columns=["attention_mask"],
        ).remove_columns(["input_ids"])

        dataset.set_format(type="numpy")

        self.batch_size = batch_size
        self.length = length
        self.epoch = 0
        self.dataset = dataset
        self.dataset_space = dataset_space

        self.shards = [self._shard(i) for i in range(self.batch_size)]

    def _shard(self, i):
        shard = iter(self.dataset.shard(num_shards=self.batch_size, index=i))
        zeros = {
            k: np.zeros((self.length, *space.shape))
            for k, space in self.dataset_space.items()
            if not k.startswith("log_") and k != "token"
        }
        while True:
            try:
                batch = next(shard)
            except StopIteration:
                if i == 0:
                    self.epoch += 1
                    print("One epoch done, re-shuffling.")
                # シャードが終わったら再シャッフルして継続
                shard = iter(self.dataset.shard(num_shards=self.batch_size, index=i).shuffle())
                batch = next(shard)
            batch.update(zeros)
            yield batch

    def __iter__(self):
        return self

    def __next__(self):
        batch = [next(shard) for shard in self.shards]
        batch = {k: np.stack([b[k] for b in batch], axis=0)
                 for k, v in batch[0].items()}
        return batch

