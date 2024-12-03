import os
import zipfile
from tqdm import tqdm  # Adicionando a biblioteca tqdm para a barra de progresso
from concurrent.futures import ThreadPoolExecutor
import rarfile

# Zip just txt files
def zip_git_files(folder_path, zip_filename):
    total_files = sum([len(files) for r, d, files in os.walk(folder_path) if any(f in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt'] for f in files)])

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Compressão (Git files)", unit="arquivo") as pbar:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt']:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))
                        pbar.update(1)

# Função para zipar apenas o conteúdo de gml e prop, ignorando os arquivos .txt com barra de progresso
def zip_gml_prop(folder_path, zip_filename):
    total_files = sum([len(files) for r, d, files in os.walk(folder_path) if not any(f in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt'] for f in files)])

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Compressão (gml e prop)", unit="arquivo") as pbar:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file not in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt']:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))
                        pbar.update(1)

def zip_files_excluding_gml_content(folder_path, zip_filename):
    # Conta o total de arquivos que não estão em subdiretórios 'gml' ou são os arquivos excluídos
    total_files = sum(
        [
            1
            for root, dirs, files in os.walk(folder_path)
            for file in files
            if not (os.path.basename(root) == 'gml')  # Ignora conteúdo de subdiretórios 'gml'
            or file in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt']  # Sempre inclui esses
        ]
    )

    # Cria o arquivo ZIP e inicia o processo de compactação
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Compressão (excluindo conteúdo de gml)", unit="arquivo") as pbar:
            for root, dirs, files in os.walk(folder_path):
                # Ignora conteúdo de subdiretórios 'gml', mas mantém os arquivos listados
                if os.path.basename(root) == 'gml':
                    # Mantém arquivos específicos nos subdiretórios 'gml'
                    for file in files:
                        if file in ['filenames.txt', 'properties_set.txt', 'time_process_seconds.txt']:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, folder_path))
                            pbar.update(1)
                    continue

                for file in files:
                    # Compacta todos os outros arquivos fora dos subdiretórios 'gml'
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
                    pbar.update(1)

# Função para extrair apenas o conteúdo da pasta "prop" com barra de progresso
def extract_prop_from_zip(zip_filename, extract_dir):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        prop_files = [f for f in zipf.namelist() if 'prop/' in f and not f.endswith('/')]
        with tqdm(total=len(prop_files), desc="Extração (prop)", unit="arquivo") as pbar:
            for file in prop_files:
                zipf.extract(file, extract_dir)
                pbar.update(1)

# Função de decompression com multithreading
def decompress_prop(zip_filename, extract_dir):
    with ThreadPoolExecutor() as executor:
        executor.submit(extract_prop_from_zip, zip_filename, extract_dir)


# Função para apagar todos os arquivos dentro de pastas gml, mantendo a estrutura de pastas
def delete_files_in_gml_folders(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # Verifica se a pasta atual é uma pasta "gml"
        if os.path.basename(root) == 'gml':
            print(f"Removendo arquivos dentro da pasta: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Apaga o arquivo
                print(f"Arquivo removido: {file_path}")
    print("Todos os arquivos dentro das pastas 'gml' foram removidos.")

# Função para extrair apenas o conteúdo da pasta "prop" com barra de progresso
def extract_prop_from_rar(rar_filename, extract_dir):
    with rarfile.RarFile(rar_filename, 'r') as rarf:
        prop_files = [f for f in rarf.namelist() if 'prop/' in f and not f.endswith('/')]
        with tqdm(total=len(prop_files), desc="Extração (prop)", unit="arquivo") as pbar:
            for file in prop_files:
                rarf.extract(file, extract_dir)
                pbar.update(1)
