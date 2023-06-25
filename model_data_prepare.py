# import library yang dibutuhkan
import argparse  # Untuk mem-parsing argumen baris perintah
import json  # Untuk membaca file JSON
import os  # Untuk operasi pada sistem berkas
from os.path import join  # Untuk menggabungkan path file
import sys  # Untuk interaksi dengan sistem
import matplotlib.image  # Untuk operasi pada gambar
from tqdm import tqdm  # Untuk menampilkan progress bar pada iterasi
import nni  # Library untuk eksperimen otomatis dan hyperparameter tuning

import torch  # Library utama untuk pemrosesan tensor dan deep learning
import torch.utils.data as data  # Modul untuk memudahkan penggunaan data dalam torch
import torchvision.utils as vutils  # Utility untuk visualisasi data torchvision
import torch.nn.functional as F  # Modul yang menyediakan fungsi-fungsi non-linear dalam torch

from data import CelebA  # Modul data yang telah didefinisikan sendiri
import attacks  # Modul serangan yang telah didefinisikan sendiri

from AttGAN.attgan import AttGAN  # Kode utama untuk model AttGAN
from AttGAN.data import check_attribute_conflict  # Fungsi untuk memeriksa konflik atribut
from AttGAN.helpers import Progressbar  # Kelas bantu untuk menampilkan progress bar
from AttGAN.utils import find_model  # Fungsi untuk mencari model yang tersimpan
import AttGAN.attacks as attgan_attack  # Modul serangan AttGAN
from stargan.solver import Solver  # Kode utama untuk model StarGAN
from AttentionGAN.AttentionGAN_v1_multi.solver import Solver as AttentionGANSolver  # Kode utama untuk model AttentionGAN
from HiSD.inference import prepare_HiSD  # Fungsi untuk mempersiapkan model HiSD

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
   # Membaca file JSON dan mengembalikan argumen yang di-parse
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# Inisialisasi AttGAN
def init_attGAN(args_attack):
    # Membaca file setting AttGAN dan mengembalikan model AttGAN yang telah di-load dan di-set ke mode evaluasi
    with open(join('.\AttGAN\output', args_attack.AttGAN.attgan_experiment_name, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    args.test_int = args_attack.AttGAN.attgan_test_int
    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.load_epoch = args_attack.AttGAN.attgan_load_epoch
    args.multi_gpu = args_attack.AttGAN.attgan_multi_gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)

    # Inisialisasi model AttGAN dan memuat checkpoint
    attgan = AttGAN(args)
    attgan.load(find_model(join('.\AttGAN\output', args.experiment_name, 'checkpoint'), args.load_epoch))
    attgan.eval()
    return attgan, args

# Inisialisasi StarGAN
def init_stargan(args_attack, test_dataloader):
    # Inisialisasi solver StarGAN dengan dataloader pengujian
    return Solver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.stargan)

# Inisialisasi AttentionGAN
def init_attentiongan(args_attack, test_dataloader):
    # Inisialisasi solver AttentionGAN dengan dataloader pengujian
    return AttentionGANSolver(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.AttentionGAN)

# Inisialisasi data serangan
def init_attack_data(args_attack, attgan_args):
    # Membuat dataset pengujian untuk serangan berdasarkan konfigurasi yang diberikan
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args_attack.global_settings.batch_size, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

# Inisialisasi data inferensi
def init_inference_data(args_attack, attgan_args):
    # Membuat dataset pengujian untuk serangan berdasarkan konfigurasi yang diberikan
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

def prepare():
    # Persiapan model deepfake dan konfigurasinya
    args_attack = parse()
    attgan, attgan_args = init_attGAN(args_attack)
    attack_dataloader = init_attack_data(args_attack, attgan_args)
    test_dataloader = init_inference_data(args_attack, attgan_args)
    solver = init_stargan(args_attack, test_dataloader)
    solver.restore_model(solver.test_iters)
    attentiongan_solver = init_attentiongan(args_attack, test_dataloader)
    attentiongan_solver.restore_model(attentiongan_solver.test_iters)
    transform, F, T, G, E, reference, gen_models = prepare_HiSD()
    print("Finished deepfake models initialization!")
    return attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models


if __name__=="__main__":
    # Panggil fungsi prepare untuk menginisialisasi model deepfake dan konfigurasinya
    prepare()