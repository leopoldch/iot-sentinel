import kagglehub
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

path_edge = kagglehub.dataset_download(
    "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
)

print(f"Téléchargé dans : {path_edge}")
