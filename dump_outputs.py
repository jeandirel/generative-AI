import json

file_path = r"c:\Users\mon pc\Downloads\PGE5\Generative AI\llm_finetune (1).ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open("dump_outputs.txt", "w", encoding="utf-8") as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            f.write(f"--- Cell {i} ---\n")
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'text' in out:
                    f.write("".join(out['text']) + "\n")
                elif 'data' in out and 'text/plain' in out['data']:
                    f.write("".join(out['data']['text/plain']) + "\n")
                elif 'name' in out and out['name'] == 'stdout':
                    f.write("".join(out['text']) + "\n")
            f.write("\n")
