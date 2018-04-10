import os
import urllib.request

WEIGHTS_URL = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy'
TRAINING_URL = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p'

def download(url, filename):
    print('Downloading {}...'.format(filename))
    urllib.request.urlretrieve(url, filename)
    filestat = os.stat(filename)
    size = filestat.st_size
    print('Successfully downloaded {} ({} bytes)'.format(filename, size))
    return filename

def get_data():
    train_file = 'train.p'
    weights_file = 'bvlc-alexnet.npy'

    if not os.path.isfile(train_file):
        download(TRAINING_URL, train_file)
    
    if not os.path.isfile(weights_file):
        download(WEIGHTS_URL, weights_file)