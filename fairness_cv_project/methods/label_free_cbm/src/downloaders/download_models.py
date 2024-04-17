"""
This file downloads the models used in the paper from the OpenAI website.
"""
import hashlib
import os
import urllib
import warnings
import logging
from typing import Any, Union, List

import torch
import torchvision

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B-32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B-16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L-14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L-14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}



def download_all_models(root: Path = Path.cwd()):
    root = Path.cwd()
    folder_save = root / "clip" / 'models' 
    folder_save.mkdir(parents=True, exist_ok=True)
    for model_name, url in _MODELS.items():
        save_path = folder_save / f"{model_name}.pt"
        _download_and_save(url, save_path)


def _download_and_save(url, save_path):
    expected_sha256 = url.split("/")[-2] 

    if os.path.isfile(save_path):
        if hashlib.sha256(open(save_path, "rb").read()).hexdigest() == expected_sha256:
            return save_path
        else:
            warnings.warn(f"{save_path} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(save_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(save_path, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")
    
    logging.info(f"Model downloaded to {save_path}")

    return

def download_resnet18_cub(save_path=Path.cwd() / 'saved_models' / 'resnet18_cub.pt'):
    """
    Download the pretrained ResNet18 model from PyTorch Computer Vision Models.
    """
    target_model = ptcv_get_model("resnet18_cub", pretrained=True)
    target_model.eval()
    torch.save(target_model, save_path)

"""
Download SentenceTransformer from HuggingFace.
"""
def download_sentence_transformer(save_path=Path.cwd() / 'saved_models' / 'sentence_transformer.pt'):
    target_model = SentenceTransformer('all-mpnet-base-v2')
    target_model.eval()
    torch.save(target_model, save_path)

def download_alexnet(save_path=Path.cwd() / 'saved_models' / 'alexnet.pt'):

    # Download the pre-trained AlexNet model
    model = torchvision.models.alexnet(pretrained=True)

    # Save the model to a file
    torch.save(model.state_dict(), save_path)
    

if __name__ == "__main__":
    # download_resnet18_cub()
    # download_all_models()
    download_sentence_transformer()
    # download_alexnet()