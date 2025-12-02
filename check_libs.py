import importlib.util

libs = ['markdown', 'pdfkit', 'weasyprint', 'xhtml2pdf']
available = []
for lib in libs:
    if importlib.util.find_spec(lib) is not None:
        available.append(lib)

print("Available libraries:", available)
