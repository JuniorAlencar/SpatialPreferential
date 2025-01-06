from src.googledriveFunctions import *

if __name__ == "__main__":
    #folder_to_compress = "../../data"
    #zip_destination = os.path.expanduser("~/Documents/N_10000.zip")

    # Baixar o arquivo do Google Drive
    #download_file_from_drive(file_id, zip_destination)
    folder_path = "../../data_3"
    zip_filename = "/home/junior/Downloads/N_320000_n_fix_1.zip"
    extract_prop_from_zip(zip_filename, folder_path)
    #zip_files_excluding_gml_content(folder_path, zip_filename)
        
    
