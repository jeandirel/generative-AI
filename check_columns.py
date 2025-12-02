from datasets import load_dataset

try:
    ds = load_dataset("Trelis/tiny-shakespeare", split="train")
    print("Column names:", ds.column_names)
    print("First example:", ds[0])
except Exception as e:
    print(f"Error loading dataset: {e}")
