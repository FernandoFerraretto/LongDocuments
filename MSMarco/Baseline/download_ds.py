import requests
import tarfile

url = "https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path="./dataset_msmarco")
