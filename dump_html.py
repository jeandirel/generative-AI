import json

file_path = r"c:\Users\mon pc\Downloads\PGE5\Generative AI\llm_finetune (1).ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open("dump_html.txt", "w", encoding="utf-8") as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'data' in out and 'text/html' in out['data']:
                    f.write(f"--- Cell {i} HTML ---\n")
                    f.write("".join(out['data']['text/html']) + "\n")
