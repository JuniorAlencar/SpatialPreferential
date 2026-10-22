from pathlib import Path

from src.processing_data import processing_all_data

# Resolve a partir deste arquivo, não do diretório em que o comando foi chamado.
folder_data = Path(__file__).resolve().parents[1] / "Data"
processing_all_data(folder_data)
