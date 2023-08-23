from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
import cv2
from tqdm import tqdm
 
class DatasetGenerator(Dataset):
    
    def __init__(self, root_dir, markup, transform):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_side_size = 512

        self.markup = markup

        self.markup_keys = list(self.markup.keys())
        self.markup_values = list(self.markup.values())

        self.images = []

        print("Loading images: ")
        for markup_key in tqdm(self.markup_keys):
            img_name = self.root_dir / markup_key
            self.images.append(cv2.imread(str(img_name)))
            
        
    def __len__(self):
        return len(self.markup.keys())

    def _get_mask(self, image, alv_r_percent, h_shift, v_shift):
        x_size = image.shape[1]
        y_size = image.shape[0]

        r = int(alv_r_percent * min(x_size, y_size) / 2)
        mask_x = range(x_size)
        mask_x = np.tile(mask_x, (y_size, 1))
        mask_y = np.array([range(y_size)]).transpose()
        mask_y = np.tile(mask_y, (1, x_size))

        c0 = y_size / 2 + v_shift
        c1 = x_size / 2 + h_shift
        mask_ids = np.where(((mask_x - c1)**2 + \
                            (mask_y - c0)**2 > r**2) |
                            (mask_x >= x_size) | 
                            (mask_y >= y_size))
        return mask_ids

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        img_name = self.markup_keys[idx]
        img = np.copy(self.images[idx])

        mask = self._get_mask(img, 0.95, 0, 0)
        img[mask] = 0

        img = cv2.resize(img, (self.img_side_size, self.img_side_size))

        sample["image"] = img

        sample["name"] = img_name
        sample["agg_type"] = np.array(self.markup_values[idx]["gt_result"], dtype = np.float32)
        sample["reagent"] = self.markup_values[idx]["reagent"]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt, name, reagent = sample['image'], sample['agg_type'], sample['name'], sample['reagent']
        image = image.transpose((2, 0, 1))

        ret_dict = {'image': torch.from_numpy(image),
                    'agg_type': torch.from_numpy(gt),
                    "name": name,
                    'reagent': reagent}
        return ret_dict