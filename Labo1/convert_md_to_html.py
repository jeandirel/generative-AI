import markdown
import os

# Configuration
input_file = "Rapport_Labo1_Template.md"
output_file = "Rapport_Labo1.html"

# CSS for the report
css = """
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 210mm;
        margin: 0 auto;
        padding: 20mm;
        background-color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    h1 {
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
    }
    h2 {
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
        margin-top: 30px;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #f8f9fa;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    blockquote {
        border-left: 4px solid #3498db;
        margin: 0;
        padding-left: 15px;
        color: #555;
        background-color: #f0f7fb;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: Consolas, Monaco, 'Andale Mono', monospace;
    }
    @media print {
        body {
            padding: 0;
            max-width: 100%;
        }
        @page {
            margin: 20mm;
        }
    }
</style>
"""

# Read Markdown
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to HTML
html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

# Create full HTML document
full_html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Labo 1</title>
    {css}
</head>
<body>
    {html_content}
</body>
</html>
"""

# Write HTML
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Successfully converted {input_file} to {output_file}")
