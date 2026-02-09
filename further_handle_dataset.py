from datasets import Dataset
import gzip
import json
import tqdm
# print(formatted[0])

with open("../nfs_folder/data_entries/multi_IRs_train_dataset.jsonl", "r") as f:
    train_dataset=[json.loads(line) for line in tqdm.tqdm(f)]
print(f"download done. len(train_dataset): {len(train_dataset)})")
formatted = [
    {
        "messages": [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["label_with_CoT"]}
        ]
    }
    for entry in train_dataset
]
print("format done.")
with open("../nfs_folder/data_entries/train_dataset_with_CoT_in_chattemplate.jsonl", "w", encoding="utf-8") as f:
    for item in tqdm.tqdm(formatted):
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("write done.")

