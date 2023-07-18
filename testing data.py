import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import universal_attack_inference_one_image
import argparse  # Modul untuk mem-parsing argumen dari command line
import copy  # Modul untuk melakukan copy objek
import json  # Modul untuk membaca dan menulis file JSON
import os  # Modul untuk berinteraksi dengan sistem operasi
from os.path import join  # Fungsi untuk menggabungkan path
import sys  # Modul yang memberikan akses ke beberapa variabel dan fungsi yang memiliki hubungan erat dengan interpreter Python
import shutil  # Modul untuk operasi file dan direktori
import matplotlib.image  # Modul untuk manipulasi gambar
from tqdm import tqdm  # Modul untuk menampilkan progress bar
from PIL import Image,ImageTk  # Modul untuk operasi gambar menggunakan PIL (Python Imaging Library)
from attacks import LinfPGDAttack  # Modul dengan serangan LinfPGDAttack yang digunakan
import torch  # Library untuk komputasi numerik menggunakan tensor
import torch.utils.data as data  # Modul untuk mengatur dataset dan dataloader
import torchvision.utils as vutils  # Modul untuk operasi-utilitas pada data visual
import torch.nn.functional as F  # Modul berisi fungsi-fungsi utilitas dalam torch.nn
from torchvision import transforms  # Modul untuk transformasi data pada gambar

from AttGAN.data import check_attribute_conflict  # Fungsi untuk memeriksa konflik atribut

from data import CelebA  # Modul dengan kelas CelebA untuk memanipulasi dataset
import attacks  # Modul dengan serangan-serangan yang akan digunakan

from model_data_prepare import prepare  # Fungsi untuk mempersiapkan model dan data
from evaluate import evaluate_multiple_models  # Fungsi untuk mengevaluasi model

# Create a new Tkinter window
window = tk.Tk()

# Define a variable to hold the selected image
selected_image = None

def main():
    universal_attack_inference_one_image.


if __name__ == "__main__":
    main()

def get_image():
    global selected_image
    # Open a file dialog to choose an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])

    # Check if a file was selected
    if file_path:
        # Open the image file using PIL
        selected_image = Image.open(file_path)

        # Resize the image to fit the window if necessary
        window_width, window_height = window.winfo_width(), window.winfo_height()
        if selected_image.width > window_width or selected_image.height > window_height:
            selected_image.thumbnail((window_width, window_height))

        # Display the image in a Tkinter label
        image_label.configure(image="")
        image_label.image = ImageTk.PhotoImage(selected_image)
        image_label.configure(image=image_label.image)

# Add a button to choose an image
button_select_image = tk.Button(window, text="Select Image", command=get_image)
button_select_image.pack()

# Add a label to display the image
image_label = tk.Label(window)
image_label.pack()

# Add a button to run the attack
button_start_attack = tk.Button(window, text="Start Attack", command=run_attack)
button_start_attack.pack()

# Start the Tkinter event loop
window.mainloop()