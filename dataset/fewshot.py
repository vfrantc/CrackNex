from dataset.transform import crop, hflip, normalize

from collections import defaultdict
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


from dataset.transform import crop, hflip, normalize
from collections import defaultdict
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset


class FewShot(Dataset):
    """
    FewShot generates support-query pairs in an episodic manner,
    intended for meta-training and meta-testing paradigm.
    """

    def __init__(self, root, size, mode, shot, episode):
        super(FewShot, self).__init__()
        self.size = size
        self.mode = mode
        self.fold = 0
        self.shot = shot
        self.episode = episode

        self.img_path = os.path.join(root, 'JPEGImages')
        self.mask_path = os.path.join(root, 'SegmentationClass')
        self.id_path = os.path.join(root, 'ImageSets')

        # class 1: normal light crack; class 2: lowlight crack
        n_class = 2

        interval = n_class // 2
        if self.mode == 'train':
            # base classes = all classes - novel classes
            self.classes = set(range(interval * self.fold + 1, interval * (self.fold + 1) + 1))
        else:
            # novel classes
            self.classes = set(range(1, n_class + 1)) - set(range(interval * self.fold + 1, interval * (self.fold + 1) + 1))

        # the image ids must be stored in 'train.txt' and 'val.txt'
        with open(os.path.join(self.id_path, f'{mode}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

        self._filter_ids()
        self.cls_to_ids = self._map_cls_to_cls()

    def __getitem__(self, item):
        # the sampling strategy is based on the description in OSLSM paper

        # query id, image, mask
        if self.mode == 'train':
            id_q = random.choice(self.ids)
        else:
            id_q = self.ids[item]

        # --- Load and remap 255 -> 1 for query mask ---
        mask_q_path = os.path.join(self.mask_path, f"{id_q}.png")
        mask_q_np = np.array(Image.open(mask_q_path))
        mask_q_np[mask_q_np == 255] = 1  # <-- remap here
        mask_q = Image.fromarray(mask_q_np)

        img_q_path = os.path.join(self.img_path, f"{id_q}.jpg")
        img_q = Image.open(img_q_path).convert('RGB')

        # Choose a valid class from the query mask
        cls = random.choice(sorted(set(np.unique(mask_q)) & self.classes))

        # support ids, images, and masks
        id_s_list, img_s_list, hiseq_s_list, mask_s_list = [], [], [], []
        while True:
            # pick a support ID that has the same class
            id_s = random.choice(sorted(set(self.cls_to_ids[cls]) - {id_q} - set(id_s_list)))

            # --- Load and remap 255 -> 1 for support mask ---
            mask_s_path = os.path.join(self.mask_path, f"{id_s}.png")
            mask_s_np = np.array(Image.open(mask_s_path))
            mask_s_np[mask_s_np == 255] = 1  # <-- remap here
            mask_s = Image.fromarray(mask_s_np)

            img_s_path = os.path.join(self.img_path, f"{id_s}.jpg")
            img_s = Image.open(img_s_path).convert('RGB')

            # small objects in support images are filtered following PFENet
            if np.sum(np.array(mask_s) == cls) < 2 * 32 * 32:
                continue

            id_s_list.append(id_s)
            img_s_list.append(img_s)
            mask_s_list.append(mask_s)
            if len(id_s_list) == self.shot:
                break

        # data augmentation (train mode)
        if self.mode == 'train':
            img_q, mask_q = crop(img_q, mask_q, self.size)
            img_q, mask_q = hflip(img_q, mask_q)
            for k in range(self.shot):
                img_s_list[k], mask_s_list[k] = crop(img_s_list[k], mask_s_list[k], self.size)
                img_s_list[k], mask_s_list[k] = hflip(img_s_list[k], mask_s_list[k])

        # normalize
        img_q, hiseq_q, mask_q = normalize(img_q, mask_q)
        for k in range(self.shot):
            img_s_list[k], hiseq_s, mask_s_list[k] = normalize(img_s_list[k], mask_s_list[k])
            hiseq_s_list.append(hiseq_s)

        # filter out irrelevant classes by setting them to background (0)
        mask_q[(mask_q != cls) & (mask_q != 255)] = 0
        mask_q[mask_q == cls] = 1
        for k in range(self.shot):
            mask_s_list[k][(mask_s_list[k] != cls) & (mask_s_list[k] != 255)] = 0
            mask_s_list[k][mask_s_list[k] == cls] = 1

        return (img_s_list, hiseq_s_list, mask_s_list,
                img_q, hiseq_q, mask_q, cls, id_s_list, id_q)

    def __len__(self):
        if self.mode == 'train':
            return self.episode
        else:
            return len(self.ids)

    # remove images that do not contain any valid classes
    # and remove images whose valid objects are all small (according to PFENet)
    def _filter_ids(self):
        for i in range(len(self.ids) - 1, -1, -1):
            path_mask = os.path.join(self.mask_path, self.ids[i] + '.png')
            mask_np = np.array(Image.open(path_mask))

            # --- remap 255 -> 1 here too ---
            mask_np[mask_np == 255] = 1
            mask = Image.fromarray(mask_np)

            classes = set(np.unique(mask)) & self.classes
            if not classes:
                del self.ids[i]
                continue

            # remove images whose valid objects are all too small
            exist_large_objects = False
            for cls in classes:
                if np.sum(mask_np == cls) >= 2 * 32 * 32:
                    exist_large_objects = True
                    break
            if not exist_large_objects:
                del self.ids[i]

    # map each valid class to a list of image ids
    def _map_cls_to_cls(self):
        cls_to_ids = defaultdict(list)
        for id_ in self.ids:
            path_mask = os.path.join(self.mask_path, id_ + ".png")
            mask_np = np.array(Image.open(path_mask))

            # --- remap 255 -> 1 here too ---
            mask_np[mask_np == 255] = 1

            valid_classes = set(np.unique(mask_np)) & self.classes
            for cls in valid_classes:
                cls_to_ids[cls].append(id_)
        return cls_to_ids
