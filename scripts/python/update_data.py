from src.googledriveFunctions import *

if __name__ == "__main__":
    #folder_to_compress = "../../data"
    #authenticate_google()
    #file_id = "1ta9ztMPwA1g2S08HLXnvuYO_F4Bsqlk3"
    #zip_destination = os.path.expanduser("~/Documents/N_10000.zip")

    # Baixar o arquivo do Google Drive
    #download_file_from_drive(file_id, zip_destination)
    folder_path = "../../data"
    zip_filename = "../../data.zip"
    zip_files_excluding_gml_content(folder_path, zip_filename)
        
    
