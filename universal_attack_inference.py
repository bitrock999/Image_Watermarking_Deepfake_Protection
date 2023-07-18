import argparse  # Modul untuk mem-parsing argumen dari command line
import copy  # Modul untuk melakukan copy objek
import json  # Modul untuk membaca dan menulis file JSON
import os  # Modul untuk berinteraksi dengan sistem operasi
import shutil  # Modul untuk operasi file dan direktori
from os.path import join  # Fungsi untuk menggabungkan path
import sys  # Modul yang memberikan akses ke beberapa variabel dan fungsi yang memiliki hubungan erat dengan interpreter Python
import matplotlib.image  # Modul untuk manipulasi gambar
from tqdm import tqdm  # Modul untuk menampilkan progress bar

import torch  # Library untuk komputasi numerik menggunakan tensor
import torch.utils.data as data  # Modul untuk mengatur dataset dan dataloader
import torchvision.utils as vutils  # Modul untuk operasi-utilitas pada data visual
import torch.nn.functional as F  # Modul berisi fungsi-fungsi utilitas dalam torch.nn

from AttGAN.data import check_attribute_conflict  # Fungsi untuk memeriksa konflik atribut

from data import CelebA  # Modul dengan kelas CelebA untuk memanipulasi dataset
import attacks  # Modul dengan serangan-serangan yang akan digunakan

from model_data_prepare import prepare  # Fungsi untuk mempersiapkan model dan data
from evaluate import evaluate_multiple_models  # Fungsi untuk mengevaluasi model


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
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        
    return args_attack


args_attack = parse()
print(args_attack)

results_dir = os.path.join(args_attack.global_settings.results_path, 'results{}'.format(args_attack.attacks.momentum))
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)  # delete existing directory

os.makedirs(results_dir)  # create new directory
print("experiment dir is created")

shutil.copy('./setting.json', os.path.join(results_dir, 'setting.json'))
print("experiment config is saved")

# init attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                       epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k, a=args_attack.attacks.a, 
                                       star_factor=args_attack.attacks.star_factor, attention_factor=args_attack.attacks.attention_factor, 
                                       att_factor=args_attack.attacks.att_factor, HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks) 
                                       # Inisialisasi serangan PGD menggunakan argumen dari objek args_attack
    return pgd_attack


pgd_attack = init_Attack(args_attack)  # Inisialisasi objek serangan PGD

# Memuat model CMUA-Watermark yang sudah dilatih
if args_attack.global_settings.universal_perturbation_path:
    pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)

# Inisialisasi model yang akan diserang dan persiapan data
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

print("finished init the attacked models")  # Mencetak pesan bahwa inisialisasi model yang diserang sudah selesai

print('The size of CMUA-Watermark: ', pgd_attack.up.shape)  # Mencetak ukuran CMUA-Watermark yang digunakan dalam serangan

evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models, pgd_attack)  # Mengevaluasi model yang diserang menggunakan serangan PGD pada dataset uji
