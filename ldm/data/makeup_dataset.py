import json
import cv2
import os
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torchvision
import torch
import random

def get_tensor(normalize=True, toTensor=True, channels=3):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize and channels == 3:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    if normalize and channels == 1:
        transform_list += [torchvision.transforms.Normalize((0.5),
                                                            (0.5))]
    return torchvision.transforms.Compose(transform_list)


class MakeupDataset(Dataset):
    def __init__(self, is_train, image_path, seg_path,depth_path):
        super(MakeupDataset, self).__init__()
        self.is_train = is_train
        self.image_path = image_path
        self.seg_path = seg_path
        self.depth_path = depth_path
        self.name_list = os.listdir(image_path)

        seg_list = os.listdir(seg_path)
        depth_list = os.listdir(depth_path)
        # check segmentation maps
        for name in self.name_list:
            assert name[
                   :-4] + '.png' in seg_list, f'Please check the segmentation folder, the segmentation map of Figure {name} does not exist'
            assert name[
                   :-4] + '_3d.jpg' in depth_list, f'Please check the 3d folder, the 3d map of Figure {name} does not exist'

        self.appearance_aug = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)
        ])

        self.appearance_aug2 = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0)
        ])

        self.spatial_aug_without_elas = A.Compose([
            A.Resize(height=286, width=286),
            A.RandomCrop(height=256, width=256),
            A.Rotate(limit=45,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0,
                     p=0.5)
        ])

        self.spatial_aug_with_elas = A.Compose([
            A.Resize(height=286, width=286),
            A.RandomCrop(height=256, width=256),
            A.Rotate(
                limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5),
            A.ElasticTransform(
                alpha=120,
                sigma=10,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8
            )
        ])

    def __len__(self):
        if not self.is_train:
            return 1000
        return len(self.name_list)

    def get_only_face_seg(self, seg):
        face_seg = np.ones_like(seg)
        face_seg[np.where(seg >= 13)] = 0
        face_seg[np.where(seg == 0)] = 0
        face_seg[np.where(seg == 3)] = 0
        face_seg[np.where(seg == 4)] = 0
        face_seg[np.where(seg == 5)] = 0
        face_seg[np.where(seg == 10)] = 0
        face_seg[np.where(seg == 17)] = 1 #neck
        # face_seg[np.where(seg == 8)] = 0 #ear
        # face_seg[np.where(seg == 9)] = 0 #ear
        return face_seg

    def get_only_neck_ears_seg(self, seg):
        neck_ears_seg = np.zeros_like(seg)
        neck_ears_seg[np.where(seg == 8)] = 1
        neck_ears_seg[np.where(seg == 9)] = 1
        neck_ears_seg[np.where(seg == 17)] = 1
        return neck_ears_seg


    def seg2onehot(self, seg, channels=19):
        out = np.zeros(shape=(seg.shape[0], seg.shape[0], channels))
        for i in range(channels):
            out[:, :, i][np.where(seg == i)] = 1
        return out

    def __getitem__(self, idx):
        if self.is_train:
            name = self.name_list[idx]

            real_ref_idx=random.randint(0,len(self.name_list)-1)
            real_ref_name=self.name_list[real_ref_idx]

            real_ref_image = cv2.imread(os.path.join(self.image_path, real_ref_name))
            real_ref_image = cv2.cvtColor(real_ref_image, cv2.COLOR_BGR2RGB)
            real_ref_seg = cv2.imread(os.path.join(self.seg_path, real_ref_name[:-4] + '.png'), flags=cv2.IMREAD_GRAYSCALE)
            real_ref_dict = self.spatial_aug_without_elas(image=real_ref_image, mask=real_ref_seg)

            real_ref_image = real_ref_dict['image']
            real_ref_seg = np.round(real_ref_dict['mask'])

            real_ref_face_seg = self.get_only_face_seg(real_ref_seg)
            real_ref_face = real_ref_image * real_ref_face_seg[:, :, None]
            real_ref_face_seg = np.expand_dims(real_ref_face_seg, axis=2)


            original_image = cv2.imread(os.path.join(self.image_path, name))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_seg = cv2.imread(os.path.join(self.seg_path, name[:-4] + '.png'), flags=cv2.IMREAD_GRAYSCALE)
            #original_depth = cv2.imread(os.path.join(self.depth_path, name[:-4] + '_depth.jpg'))
            original_depth = cv2.imread(os.path.join(self.depth_path, name[:-4] + '_3d.jpg'))

            appearance_change_dict = self.appearance_aug(image=original_image, mask=original_seg)
            appearance_change_image = appearance_change_dict['image']
            original_seg = appearance_change_dict['mask']

            appearance_change_image_depth=np.concatenate([appearance_change_image,original_depth],axis=2)

            source_dict = self.spatial_aug_without_elas(image=appearance_change_image_depth, mask=original_seg)

            image_GT = source_dict['image'][:,:,0:3]
            source_depth = source_dict['image'][:, :, 3:6]
            # print(np.unique(image_depth))
            # print('img GT',image_GT.shape)
            source_seg = np.round(source_dict['mask'])
            source_seg_onehot=self.seg2onehot(source_seg)
            source_face_seg = self.get_only_face_seg(source_seg)
            source_neck_ears_seg = self.get_only_neck_ears_seg(source_seg)

            source_bg = image_GT * (1 - source_face_seg[:, :, None])

            source_face = image_GT * source_face_seg[:, :, None]

            source_face_seg=np.expand_dims(source_face_seg, axis=2)
            source_neck_ears_seg = np.expand_dims(source_neck_ears_seg, axis=2)

            source_face_change = self.appearance_aug2(image=source_face)['image']
            source_face_gray = cv2.cvtColor(source_face_change, cv2.COLOR_BGR2GRAY)
            source_face_gray = np.expand_dims(source_face_gray, axis=2)
            # print(source_face_gray.shape)

            ref_dict = self.spatial_aug_with_elas(image=appearance_change_image, mask=original_seg)
            ref_image = ref_dict['image']
            ref_seg = np.round(ref_dict['mask'])
            ref_seg_onehot = self.seg2onehot(ref_seg)
            ref_face_seg = self.get_only_face_seg(ref_seg)
            ref_face = ref_image * ref_face_seg[:, :, None]


            image_GT = get_tensor(normalize=True, toTensor=True)(image_GT)
            source_depth = get_tensor(normalize=True, toTensor=True)(source_depth)
            source_bg = get_tensor(normalize=True, toTensor=True)(source_bg)
            source_face_gray = get_tensor(normalize=True, toTensor=True, channels=1)(source_face_gray)
            ref_face = get_tensor(normalize=True, toTensor=True)(ref_face)
            real_ref_face = get_tensor(normalize=True, toTensor=True)(real_ref_face)
            source_seg_onehot=torch.from_numpy(np.transpose(source_seg_onehot,(2,0,1)))
            ref_seg_onehot = torch.from_numpy(np.transpose(ref_seg_onehot, (2, 0, 1)))
            source_face_seg = torch.from_numpy(np.transpose(source_face_seg, (2, 0, 1)))
            source_neck_ears_seg = torch.from_numpy(np.transpose(source_neck_ears_seg, (2, 0, 1)))
            real_ref_face_seg = torch.from_numpy(np.transpose(real_ref_face_seg, (2, 0, 1)))

            data_dict = {'image_GT': image_GT,
                         'source_depth': source_depth,
                         'source_bg': source_bg,
                         'source_face_gray': source_face_gray,
                         'ref_face': ref_face,
                         'real_ref_face': real_ref_face,
                         'source_seg_onehot':source_seg_onehot,
                         'ref_seg_onehot':ref_seg_onehot,
                         'source_face_seg': source_face_seg,
                         'source_neck_ears_seg': source_neck_ears_seg,
                         'real_ref_face_seg': real_ref_face_seg
                         }

            return data_dict
        else:
            name = self.name_list[idx]

            real_ref_idx = random.randint(0, len(self.name_list) - 1)
            real_ref_name = self.name_list[real_ref_idx]

            real_ref_image = cv2.imread(os.path.join(self.image_path, real_ref_name))
            real_ref_image = cv2.cvtColor(real_ref_image, cv2.COLOR_BGR2RGB)
            real_ref_seg = cv2.imread(os.path.join(self.seg_path, real_ref_name[:-4] + '.png'),
                                      flags=cv2.IMREAD_GRAYSCALE)
            real_ref_dict = self.spatial_aug_without_elas(image=real_ref_image, mask=real_ref_seg)

            real_ref_image = real_ref_dict['image']
            real_ref_seg = np.round(real_ref_dict['mask'])

            real_ref_face_seg = self.get_only_face_seg(real_ref_seg)
            real_ref_face = real_ref_image * real_ref_face_seg[:, :, None]
            real_ref_face_seg = np.expand_dims(real_ref_face_seg, axis=2)

            original_image = cv2.imread(os.path.join(self.image_path, name))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_seg = cv2.imread(os.path.join(self.seg_path, name[:-4] + '.png'), flags=cv2.IMREAD_GRAYSCALE)
            #original_depth = cv2.imread(os.path.join(self.depth_path, name[:-4] + '_depth.jpg'))
            original_depth = cv2.imread(os.path.join(self.depth_path, name[:-4] + '_3d.jpg'))

            appearance_change_dict = self.appearance_aug(image=original_image, mask=original_seg)
            appearance_change_image = appearance_change_dict['image']
            original_seg = appearance_change_dict['mask']

            appearance_change_image_depth = np.concatenate([appearance_change_image, original_depth], axis=2)

            source_dict = self.spatial_aug_without_elas(image=appearance_change_image_depth, mask=original_seg)

            image_GT = source_dict['image'][:, :, 0:3]
            source_depth = source_dict['image'][:, :, 3:6]
            # print(np.unique(image_depth))
            # print('img GT',image_GT.shape)
            source_seg = np.round(source_dict['mask'])
            source_seg_onehot = self.seg2onehot(source_seg)
            source_face_seg = self.get_only_face_seg(source_seg)
            source_neck_ears_seg = self.get_only_neck_ears_seg(source_seg)

            source_bg = image_GT * (1 - source_face_seg[:, :, None])

            source_face = image_GT * source_face_seg[:, :, None]

            source_face_seg = np.expand_dims(source_face_seg, axis=2)
            source_neck_ears_seg = np.expand_dims(source_neck_ears_seg, axis=2)

            source_face_change = self.appearance_aug2(image=source_face)['image']
            source_face_gray = cv2.cvtColor(source_face_change, cv2.COLOR_BGR2GRAY)
            source_face_gray = np.expand_dims(source_face_gray, axis=2)
            # print(source_face_gray.shape)

            ref_dict = self.spatial_aug_with_elas(image=appearance_change_image, mask=original_seg)
            ref_image = ref_dict['image']
            ref_seg = np.round(ref_dict['mask'])
            ref_seg_onehot = self.seg2onehot(ref_seg)
            ref_face_seg = self.get_only_face_seg(ref_seg)
            ref_face = ref_image * ref_face_seg[:, :, None]

            image_GT = get_tensor(normalize=True, toTensor=True)(image_GT)
            source_depth = get_tensor(normalize=True, toTensor=True)(source_depth)
            source_bg = get_tensor(normalize=True, toTensor=True)(source_bg)
            source_face_gray = get_tensor(normalize=True, toTensor=True, channels=1)(source_face_gray)
            ref_face = get_tensor(normalize=True, toTensor=True)(ref_face)
            real_ref_face = get_tensor(normalize=True, toTensor=True)(real_ref_face)
            source_seg_onehot = torch.from_numpy(np.transpose(source_seg_onehot, (2, 0, 1)))
            ref_seg_onehot = torch.from_numpy(np.transpose(ref_seg_onehot, (2, 0, 1)))
            source_face_seg = torch.from_numpy(np.transpose(source_face_seg, (2, 0, 1)))
            source_neck_ears_seg = torch.from_numpy(np.transpose(source_neck_ears_seg, (2, 0, 1)))
            real_ref_face_seg = torch.from_numpy(np.transpose(real_ref_face_seg, (2, 0, 1)))


            data_dict = {'image_GT': image_GT,
                         'source_depth': source_depth,
                         'source_bg': source_bg,
                         'source_face_gray': source_face_gray,
                         'ref_face': ref_face,
                         'real_ref_face': real_ref_face,
                         'source_seg_onehot': source_seg_onehot,
                         'ref_seg_onehot': ref_seg_onehot,
                         'source_face_seg': source_face_seg,
                         'source_neck_ears_seg': source_neck_ears_seg,
                         'real_ref_face_seg': real_ref_face_seg
                         }

            return data_dict


class MakeupDatasetTest(Dataset):
    def __init__(self, mode, source_image_path, source_seg_path, source_depth_path, ref_image_path, ref_seg_path):
        super(MakeupDatasetTest, self).__init__()
        self.mode = mode
        self.source_image_path = source_image_path
        self.source_seg_path = source_seg_path
        self.source_depth_path = source_depth_path
        self.ref_image_path = ref_image_path
        self.ref_seg_path = ref_seg_path

        self.source_name_list = os.listdir(source_image_path)
        self.ref_name_list = os.listdir(ref_image_path)

        self.source_name_list=np.sort(self.source_name_list)
        self.ref_name_list = np.sort(self.ref_name_list)

        source_seg_list = os.listdir(source_seg_path)
        source_depth_list = os.listdir(source_depth_path)
        ref_seg_list = os.listdir(ref_seg_path)

        self.transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=256, width=256)
        ])

        # check segmentation maps
        for name in self.source_name_list:
            assert name[
                   :-4] + '.png' in source_seg_list, f'Please check the source segmentation folder, the segmentation map of Figure {name} does not exist'
            assert name[
                   :-4] + '_3d.jpg' in source_depth_list, f'Please check the source 3d folder, the depth map of Figure {name} does not exist'

        for name in self.ref_name_list:
            assert name[
                   :-4] + '.png' in ref_seg_list, f'Please check the ref segmentation folder, the segmentation map of Figure {name} does not exist'

    def __len__(self):
        if self.mode=='test_pair':
            return len(self.source_name_list)*len(self.ref_name_list)
        else:
            return len(self.source_name_list)

    def get_only_face_seg(self, seg):
        face_seg = np.ones_like(seg)
        face_seg[np.where(seg >= 13)] = 0
        face_seg[np.where(seg == 0)] = 0
        face_seg[np.where(seg == 3)] = 0
        face_seg[np.where(seg == 4)] = 0
        face_seg[np.where(seg == 5)] = 0
        face_seg[np.where(seg == 10)] = 0
        face_seg[np.where(seg == 17)] = 1
        return face_seg

    def get_only_neck_ears_seg(self, seg):
        neck_ears_seg = np.zeros_like(seg)
        neck_ears_seg[np.where(seg == 8)] = 1
        neck_ears_seg[np.where(seg == 9)] = 1
        neck_ears_seg[np.where(seg == 17)] = 1
        return neck_ears_seg

    def seg2onehot(self, seg, channels=19):
        out = np.zeros(shape=(seg.shape[0], seg.shape[0], channels))
        for i in range(channels):
            out[:, :, i][np.where(seg == i)] = 1
        return out

    def __getitem__(self, idx):
        if self.mode=='test_pair':
            source_index = idx % len(self.source_name_list)
            ref_index = idx // len(self.source_name_list)

            source_name=self.source_name_list[source_index]
            ref_name = self.ref_name_list[ref_index]

            source_image = cv2.imread(os.path.join(self.source_image_path, source_name))
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_depth = cv2.imread(os.path.join(self.source_depth_path, source_name[:-4] + '_3d.jpg'))
            source_seg = cv2.imread(os.path.join(self.source_seg_path, source_name[:-4] + '.png'), flags=cv2.IMREAD_GRAYSCALE)

            source_image_depth = np.concatenate([source_image, source_depth], axis=2)

            source_dict=self.transforms(image=source_image_depth, mask=source_seg)
            source_image=source_dict['image'][:,:,0:3]
            source_depth = source_dict['image'][:, :, 3:6]
            source_seg = source_dict['mask']

            source_seg_onehot = self.seg2onehot(source_seg)
            source_face_seg = self.get_only_face_seg(source_seg)
            source_neck_ears_seg = self.get_only_neck_ears_seg(source_seg)

            source_bg = source_image * (1 - source_face_seg[:, :, None])
            source_face = source_image * source_face_seg[:, :, None]
            source_face_gray = cv2.cvtColor(source_face, cv2.COLOR_BGR2GRAY)
            source_face_gray = np.expand_dims(source_face_gray, axis=2)

            source_face_seg = np.expand_dims(source_face_seg, axis=2)
            source_neck_ears_seg = np.expand_dims(source_neck_ears_seg, axis=2)

            ref_image = cv2.imread(os.path.join(self.ref_image_path, ref_name))
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_seg = cv2.imread(os.path.join(self.ref_seg_path, ref_name[:-4] + '.png'),
                                    flags=cv2.IMREAD_GRAYSCALE)
            ref_dict = self.transforms(image=ref_image, mask=ref_seg)
            ref_image = ref_dict['image']
            ref_seg = ref_dict['mask']

            ref_seg_onehot = self.seg2onehot(ref_seg)
            ref_face_seg = self.get_only_face_seg(ref_seg)
            ref_face = ref_image * ref_face_seg[:, :, None]


            source_depth = get_tensor(normalize=True, toTensor=True)(source_depth)
            source_bg = get_tensor(normalize=True, toTensor=True)(source_bg)
            source_face_gray = get_tensor(normalize=True, toTensor=True, channels=1)(source_face_gray)
            ref_face = get_tensor(normalize=True, toTensor=True)(ref_face)

            source_image = get_tensor(normalize=True, toTensor=True)(source_image)
            ref_image = get_tensor(normalize=True, toTensor=True)(ref_image)

            source_seg_onehot = torch.from_numpy(np.transpose(source_seg_onehot, (2, 0, 1)))
            ref_seg_onehot = torch.from_numpy(np.transpose(ref_seg_onehot, (2, 0, 1)))

            source_face_seg = torch.from_numpy(np.transpose(source_face_seg, (2, 0, 1)))
            source_neck_ears_seg = torch.from_numpy(np.transpose(source_neck_ears_seg, (2, 0, 1)))

            data_dict = {'source_image': source_image,
                         'ref_image': ref_image,
                         'source_depth': source_depth,
                         'source_bg': source_bg,
                         'source_face_gray': source_face_gray,
                         'ref_face': ref_face,

                         'source_seg_onehot': source_seg_onehot,
                         'ref_seg_onehot': ref_seg_onehot,
                         'source_face_seg': source_face_seg,
                         'source_neck_ears_seg': source_neck_ears_seg
                         }

            return data_dict

        if self.mode=='test_1000':
            source_index = idx
            ref_index = idx

            source_name=self.source_name_list[source_index]
            ref_name = self.ref_name_list[ref_index]

            source_image = cv2.imread(os.path.join(self.source_image_path, source_name))
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_depth = cv2.imread(os.path.join(self.source_depth_path, source_name[:-4] + '_3d.jpg'))
            source_seg = cv2.imread(os.path.join(self.source_seg_path, source_name[:-4] + '.png'),
                                    flags=cv2.IMREAD_GRAYSCALE)

            source_image_depth = np.concatenate([source_image, source_depth], axis=2)

            source_dict = self.transforms(image=source_image_depth, mask=source_seg)
            source_image = source_dict['image'][:, :, 0:3]
            source_depth = source_dict['image'][:, :, 3:6]
            source_seg = source_dict['mask']

            source_seg_onehot = self.seg2onehot(source_seg)
            source_face_seg = self.get_only_face_seg(source_seg)
            source_neck_ears_seg = self.get_only_neck_ears_seg(source_seg)

            source_bg = source_image * (1 - source_face_seg[:, :, None])
            source_face = source_image * source_face_seg[:, :, None]
            source_face_gray = cv2.cvtColor(source_face, cv2.COLOR_BGR2GRAY)
            source_face_gray = np.expand_dims(source_face_gray, axis=2)

            source_face_seg = np.expand_dims(source_face_seg, axis=2)
            source_neck_ears_seg = np.expand_dims(source_neck_ears_seg, axis=2)

            ref_image = cv2.imread(os.path.join(self.ref_image_path, ref_name))
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            ref_seg = cv2.imread(os.path.join(self.ref_seg_path, ref_name[:-4] + '.png'),
                                 flags=cv2.IMREAD_GRAYSCALE)
            ref_dict = self.transforms(image=ref_image, mask=ref_seg)
            ref_image = ref_dict['image']
            ref_seg = ref_dict['mask']

            ref_seg_onehot = self.seg2onehot(ref_seg)
            ref_face_seg = self.get_only_face_seg(ref_seg)
            ref_face = ref_image * ref_face_seg[:, :, None]

            source_depth = get_tensor(normalize=True, toTensor=True)(source_depth)
            source_bg = get_tensor(normalize=True, toTensor=True)(source_bg)
            source_face_gray = get_tensor(normalize=True, toTensor=True, channels=1)(source_face_gray)
            ref_face = get_tensor(normalize=True, toTensor=True)(ref_face)

            source_image = get_tensor(normalize=True, toTensor=True)(source_image)
            ref_image = get_tensor(normalize=True, toTensor=True)(ref_image)

            source_seg_onehot = torch.from_numpy(np.transpose(source_seg_onehot, (2, 0, 1)))
            ref_seg_onehot = torch.from_numpy(np.transpose(ref_seg_onehot, (2, 0, 1)))

            source_face_seg = torch.from_numpy(np.transpose(source_face_seg, (2, 0, 1)))
            source_neck_ears_seg = torch.from_numpy(np.transpose(source_neck_ears_seg, (2, 0, 1)))

            data_dict = {'source_image': source_image,
                         'ref_image': ref_image,
                         'source_depth': source_depth,
                         'source_bg': source_bg,
                         'source_face_gray': source_face_gray,
                         'ref_face': ref_face,

                         'source_seg_onehot': source_seg_onehot,
                         'ref_seg_onehot': ref_seg_onehot,
                         'source_face_seg': source_face_seg,
                         'source_neck_ears_seg': source_neck_ears_seg
                         }

            return data_dict


if __name__ == '__main__':

    # check data load
    def save_imgs(imgs, names, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for img, name in zip(imgs, names):
            img = np.array(img.cpu().detach())
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2]==1:
                # print(np.unique(img))
                cv2.imwrite(os.path.join(path, name + '.jpg'), img*255.)
            else:
                img = (img + 1.) * 127.5
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path, name + '.jpg'), img)


    image_path = './MakeupLDM/load_test/images'
    seg_path = './MakeupLDM/load_test/segs'
    depth_path = './MakeupLDM/load_test/depth'

    save_item_name = ['image_GT', 'source_depth','source_bg', 'source_face_gray', 'ref_face','source_face_seg','source_neck_ears_seg']

    dataset = MakeupDataset(True, image_path, seg_path,depth_path)
    print(len(dataset))
    number = 4
    for i in range(number):
        index = np.random.randint(0, len(dataset) - 1)
        data_dict = dataset[index]
        imgs = [data_dict[x] for x in save_item_name]
        names = [str(i) + '_' + x for x in save_item_name]
        save_imgs(imgs, names, path='../../makeup_dataset_test')
