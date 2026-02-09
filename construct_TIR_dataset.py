import json
import tqdm

if __name__=='__main__':
  with open("../nfs_folder/data_entries/multi_IRs_train_dataset.jsonl","r") as f:
    dataset=[json.loads(line) for line in f]
  print(len(dataset))
  new_dataset=[]
  for data in tqdm.tqdm(dataset):
    original_prompt=data['prompt']
    new_prompt=original_prompt.replace(data['original_IR'], data['original_TIR'])
    new_dataset.append({"prompt":new_prompt, "label":str(data["TIR_label"])})
  formatted = [
    {
        "messages": [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["label"]}
        ]
    }
    for entry in new_dataset
  ]
  print("format done.")
  with open("../nfs_folder/data_entries/TIR_train.jsonl", "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
  print("write done.")

