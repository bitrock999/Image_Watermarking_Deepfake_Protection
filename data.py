from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import cv2
import numpy as np
import json

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, stargan_selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.stargan_selected_attrs = stargan_selected_attrs

        # Baca file atribut CelebA
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]

        # Baca informasi gambar dan label dari file atribut
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        if mode == 'train':
            self.images = images[:500]  # Mengambil 500 gambar pertama untuk mode train
            self.labels = labels[:500]  # Mengambil 500 label pertama untuk mode train
        if mode == 'valid':
            self.images = images[128:872]  # Mengambil gambar ke-128 hingga ke-871 untuk mode valid
            self.labels = labels[128:872]  # Mengambil label ke-128 hingga ke-871 untuk mode valid
        if mode == 'test':
            self.images = images[500:]  # Mengambil gambar setelah 500 untuk mode test
            self.labels = labels[500:]  # Mengambil label setelah 500 untuk mode test
        
        # Definisikan transformasi gambar
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),  # Memotong gambar menjadi bentuk persegi dengan mempertahankan pusatnya
            transforms.Resize(image_size),  # Mengubah ukuran gambar menjadi ukuran yang diinginkan
            transforms.ToTensor(),  # Mengubah gambar menjadi tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalisasi nilai piksel gambar
        ])
                                       
        self.length = len(self.images)

        # stargan
        self.attr2idx = {}
        self.idx2attr = {}
        self.test_dataset = []
        self.preprocess()

    def preprocess(self):
        """Mempersiapkan file atribut CelebA."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()

        # Buat kamus untuk menghubungkan atribut dengan indeksnya
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[500:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.stargan_selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            # Tambahkan data gambar dan label ke dataset uji
            self.test_dataset.append([filename, label])
        print('Selesai mempersiapkan dataset CelebA...')

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.images[index])
        filename = os.path.basename(img_path)
        # Ubah format nama file sesuai format yang diinginkan
        new_filename = 'frame_det_00_' + filename
        new_filename = new_filename.replace('\\\\', '\\')
        new_img_path = os.path.join(os.path.dirname(img_path), new_filename)
        new_img_path = new_img_path.replace('\\', '/')

        img = self.tf(Image.open(new_img_path))
        att = torch.tensor((self.labels[index] + 1) // 2)
        filename, label = self.test_dataset[index]

        return img, att, torch.FloatTensor(label)
        
    def __len__(self):
        return self.length
