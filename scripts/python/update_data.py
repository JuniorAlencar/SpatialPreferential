from src.googledriveFunctions import *

import os

def is_file_corrupted(file_path):
    """Verifica se um arquivo está corrompido tentando abri-lo e lendo um pequeno trecho."""
    try:
        with open(file_path, 'rb') as f:
            f.read(1024)  # Tenta ler os primeiros 1024 bytes
        return False  # O arquivo não está corrompido
    except Exception as e:
        print(f"Arquivo corrompido detectado: {file_path} -> {e}")
        return True  # O arquivo está corrompido

def clean_prop_folders(base_path):
    """Percorre todas as pastas 'prop/' e remove arquivos vazios ou corrompidos."""
    for root, _, files in os.walk(base_path):
        if "prop" in root:  # Garante que estamos apenas nas pastas 'prop/'
            for file in files:
                file_path = os.path.join(root, file)

                # Remove arquivos vazios
                if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
                    print(f"Removendo arquivo vazio: {file_path}")
                    os.remove(file_path)
                    continue  # Pula para o próximo arquivo

                # Remove arquivos corrompidos
                if is_file_corrupted(file_path):
                    print(f"Removendo arquivo corrompido: {file_path}")
                    os.remove(file_path)




if __name__ == "__main__":
    #folder_to_compress = "../../data"
    #zip_destination = os.path.expanduser("~/Documents/N_10000.zip")

    # Baixar o arquivo do Google Drive
    #download_file_from_drive(file_id, zip_destination)
    #folder_path = "../../data_3"
    #zip_filename = "/home/junior/Downloads/data_fix_33.zip"
    #extract_prop_from_zip(zip_filename, folder_path)
    extract_dir = "/home/junior/Downloads/"
    zip_filename = "/home/junior/Downloads/data_n_n.zip"
    #delete_files_in_gml_folders(folder_base)
    #extract_prop_from_zip(zip_filename, extract_dir)
    # 🛠️ Caminho base onde as pastas 'prop/' foram extraídas
    base_directory = "/home/junior/Downloads/data_n_n"  # Atualize se necessário

    # 🚀 Executar a limpeza
    clean_prop_folders(base_directory)

    print("✅ Limpeza concluída! Arquivos corrompidos e vazios foram removidos.")

    #zip_files_excluding_gml_content(folder_path, zip_filename)
        
    
