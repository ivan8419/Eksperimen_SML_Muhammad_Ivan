import nbformat
import os

# Baca file .ipynb
notebook_path = 'Eksperimen_Muhammad-Ivan.ipynb'
nb = nbformat.read(notebook_path, as_version=4)

# Clear semua output dari cell code
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

# Simpan kembali file .ipynb
nbformat.write(nb, notebook_path)

print(f"Successfully cleared all outputs from {notebook_path}")
