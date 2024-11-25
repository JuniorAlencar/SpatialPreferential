from src.googledriveFunctions import *

if __name__ == "__main__":
    #folder_to_compress = "../../data"
    #authenticate_google()
    #file_id = "1ta9ztMPwA1g2S08HLXnvuYO_F4Bsqlk3"
    #zip_destination = os.path.expanduser("~/Documents/N_10000.zip")

    # Baixar o arquivo do Google Drive
    #download_file_from_drive(file_id, zip_destination)
    lst = ["N_5000.zip","N_10000.zip", "N_20000.zip", "N_80000.zip", "data_2.zip", "N_40000.zip"]
    for ele in lst:
        extract_to_dir2 = os.path.expanduser("~/Downloads")
        zip_destination2 = os.path.expanduser(f"~/Downloads/{ele}")
        #extract_prop_from_rar(zip_destination2, extract_to_dir2)
        extract_prop_from_zip(zip_destination2, extract_to_dir2)
        
    
