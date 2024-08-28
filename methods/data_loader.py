from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
from pathlib import Path
import concurrent.futures
from torch.utils.data import TensorDataset

class ImSituLoader(data.Dataset):
    def __init__(self, balance='balanced', dataset_dir='data/datasets/imSitu/', transform_name='ResNet', transform=None, split='train', num_verbs=200, perturbation_rate=0.):
        self.balance = balance
        self.dataset_dir = dataset_dir
        self.num_verbs = num_verbs
        self.split = split
        self.perturbation = perturbation_rate
        self.image_dir = Path(dataset_dir) / 'data_processed' / Path(str(num_verbs) + '_verbs') / self.balance / split

        # Create the transform
        if transform_name == 'ResNet':
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Handle images with alpha channel
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif transform_name == 'ResNet' and split == 'train':
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),  # Handle images with alpha channel
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif transform_name == 'clip' and transform is None:
            raise ValueError('clip transform should be provided')
        elif transform_name == 'clip':
            self.data_transforms = transform
        else:
            self.data_transforms = None
            print('Not doing any preprocessing. Don\'t you need clip or ResNet?')

        self.ann_data = ImageFolder(self.image_dir)
        print('Loading data from', dataset_dir, split, len(self.ann_data), balance, 'now parallelizing')
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.image_metadata = list(executor.map(self.extract_metadata_single, range(len(self.ann_data))))
        
        if self.perturbation > 0:
            self.perturb()

    def extract_metadata_single(self, idx):
        image_path = self.ann_data.imgs[idx][0]
        verb = self.ann_data[idx][1]
        image_name = Path(image_path).name
        gender = image_name.split('.')[0].split('_')[-1]
        return {
            'verb': verb,
            'path': image_path,
            'name': image_name,
            'gender': gender
        }

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, index):
        image = self.ann_data[index][0]
        
        # Creating a verb tensor
        verb = self.image_metadata[index]['verb']

        if isinstance(verb, int) or verb.numel() == 1:
            verb_tensor = torch.zeros(self.num_verbs)
            verb_tensor[int(verb)] = 1
        else:  # Assuming verb is already a tensor
            verb_tensor = verb

        image_path = self.image_metadata[index]['path']
        image_name = self.image_metadata[index]['name']
        
        # Creating a gender tensor
        gender = self.image_metadata[index]['gender']
        gender_tensor = torch.zeros(2)
        if gender == 'male':
            gender_tensor[1] = 1
        elif gender == 'female':
            gender_tensor[0] = 1
        else:
            raise ValueError('Unknown gender')

        image = Image.open(image_path).convert('RGB')

        if self.data_transforms is not None:
            image = self.data_transforms(image)

        # [image, verb_tensor, image_path, image_name, gender_tensor]
        return image, verb_tensor, image_path, image_name, gender_tensor

    def get_genders(self):    
        genders = []
        for i in range(len(self.ann_data)):
            gender_str = self.image_metadata[i]['gender']
            gender = [1 if gender_str == 'male' else 0]
            gender_tensor = torch.zeros(2)
            gender_tensor[gender] = 1
            genders.append(gender_tensor)
        return torch.stack(genders)

    def get_verbs(self):    
        verbs = []
        for i in range(len(self.ann_data)):
            verb = self.image_metadata[i]['verb']
            if isinstance(verb, int):
                verb_tensor = torch.zeros(self.num_verbs)
                verb_tensor[verb] = 1
            else:  # Assuming verb is already a tensor
                verb_tensor = verb
            verbs.append(verb_tensor)
        return torch.stack(verbs)

    def perturb(self): # With a chance of self.perturbation, change the verb to a random other verb
        print('Perturbating', 100 * self.perturbation, '% of the verbs')
        for i in range(len(self.image_metadata)):
            if random.random() < self.perturbation:
                # Generate a number from 0 to num_verbs - 1, excluding the current verb
                new_verb = random.randint(0, self.num_verbs - 1)
                # Re-generate if the new verb is same as the old verb
                while new_verb == self.image_metadata[i]['verb']:
                    new_verb = random.randint(0, self.num_verbs - 1)
                self.image_metadata[i]['verb'] = new_verb  # Assigning the new verbs

    def change_verbs(self, predicted_verbs): 
        # Instead of storing ground truth verbs, store the given predicted verbs
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            #print(predicted_verbs)
            #print(len(predicted_verbs))
            
            for i in range(len(self.image_metadata)):
                #print('predicted_verbs[i]', predicted_verbs[i])
                futures.append(executor.submit(self._update_verb, i, predicted_verbs[i]))
            for future in futures:
                future.result()

    def _update_verb(self, index, new_verb):
        old_verb = self.image_metadata[index]['verb']
        if isinstance(new_verb, torch.Tensor):
            self.image_metadata[index]['verb'] = new_verb
        else:
            self.image_metadata[index]['verb'] = torch.tensor(new_verb)  # Convert to tensor if not already
        #print(f'Changed verb from {old_verb} to {self.image_metadata[index]["verb"]}')

    def __str__(self):
        return f'ImSituLoader with {len(self.image_metadata)} items'

    


if __name__ == '__main__':
    ImSituLoader()



class IndexedTensorDataset(TensorDataset): 
    def __getitem__(self, index): 
        val = super(IndexedTensorDataset, self).__getitem__(index)
        return val + (index,)