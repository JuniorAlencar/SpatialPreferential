from src.googledriveFunctions import *

if __name__ == "__main__":
    #folder_to_compress = "../../data"
    #authenticate_google()
    #file_id = "1ta9ztMPwA1g2S08HLXnvuYO_F4Bsqlk3"
    #zip_destination = os.path.expanduser("~/Documents/N_10000.zip")

    # Baixar o arquivo do Google Drive
    #download_file_from_drive(file_id, zip_destination)
    
    extract_to_dir2 = os.path.expanduser("~/Downloads")
    zip_destination2 = os.path.expanduser("~/Downloads/N_100000_3.zip")
    #extract_prop_from_rar(zip_destination2, extract_to_dir2)
    extract_prop_from_zip(zip_destination2, extract_to_dir2)
    #folder_path = "../../data"
    #zip_filename = "just_prop.zip"
    #zip_gml_prop(folder_path, zip_filename)

    #delete_files_in_gml_folders(folder_to_compress)
#    zip_output_filename = "base_data_1.zip"
#    google_drive_folder_id = "1CEFPFhArKMENPxD4c4j6qOZ40nN9iUmt"  # Use None para enviar para a raiz do Drive ou coloque o ID da pasta
#    compress_and_upload_git_files(folder_to_compress, google_drive_folder_id)
#    compress_and_upload_gml_prop(folder_to_compress, google_drive_folder_id)
    
    
    
    
    
    
