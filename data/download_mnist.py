import os
import urllib.request

# URLs oficiales de MNIST
DATA_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_lbl': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_lbl': 't10k-labels-idx1-ubyte.gz'
}

def download_mnist(path='data'):
    # Asegurar que el directorio existe
    if not os.path.exists(path):
        os.makedirs(path)
        
    for key, filename in FILES.items():
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Descargando {filename}...")
            url = DATA_URL + filename
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f" -> Descarga completada: {filepath}")
            except Exception as e:
                print(f"Error descargando {filename}: {e}")
        else:
            print(f"Archivo ya existe: {filename}")

if __name__ == "__main__":
    download_mnist(path='data')