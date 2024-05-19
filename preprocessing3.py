import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download

# Download stopwords jika belum diunduh
download('stopwords')

# Fungsi untuk menampilkan gambar pada Label
def show_logo(image_path, parent):
    img = tk.PhotoImage(file=image_path)
    label = tk.Label(parent, image=img, bg='lightblue')
    label.image = img  # Simpan referensi ke gambar agar tidak dihapus oleh garbage collector
    label.pack()

def import_csv():
    # Fungsi import CSV
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            # Membaca file CSV dan memuatnya ke dalam DataFrame Pandas
            df = pd.read_csv(file_path)
            # Menampilkan jumlah baris yang telah diimpor
            messagebox.showinfo("Info", f"{len(df)} baris telah diimpor dari file CSV.")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

def preprocess_text(text):
    # Fungsi pra-pemrosesan teks
    text=re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n", "",text)
    text=re.sub('\s+',' ',text)
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = text.lower()  # Konversi teks menjadi huruf kecil
    
    # Pembersihan teks menggunakan regex_pattern untuk menghapus emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"                # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Penghapusan stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_dataset(df):
    # Pra-pemrosesan teks hanya pada kolom 'full_text'
    df['Processed_Text'] = df['full_text'].apply(preprocess_text)
    # Hanya menyimpan kolom 'Processed_Text' dalam dataframe hasil preprocessing
    df_processed = df[['Processed_Text']].copy()
    return df_processed

def save_preprocessed_data(df):
    # Membuka dialog untuk menyimpan file CSV
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            # Menyimpan dataframe ke dalam file CSV
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Info", "Data berhasil disimpan ke dalam file CSV.")
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat menyimpan data: {str(e)}")

def show_raw_data(df):
    # Menampilkan data mentah
    top = tk.Toplevel(root)
    top.title("Raw Data")
    top.geometry("800x400")  # Lebarkan jendela Toplevel

    # Buat frame untuk menampilkan tabel
    frame = tk.Frame(top)
    frame.pack(fill=tk.BOTH, expand=True)

    # Buat tabel dengan Pandas DataFrame
    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
    for column in df.columns:
        tree.heading(column, text=column)
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Buat scrollbar
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)

def show_preprocessed_data(df):
    # Menampilkan data yang telah diproses
    top = tk.Toplevel(root)
    top.title("Preprocessed Data")
    top.geometry("800x400")  # Lebarkan jendela Toplevel

    # Buat frame untuk menampilkan tabel
    frame = tk.Frame(top)
    frame.pack(fill=tk.BOTH, expand=True)

    # Buat tabel dengan Pandas DataFrame
    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
    for column in df.columns:
        tree.heading(column, text=column)
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Buat scrollbar
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)

def preprocessing_button_click():
    # Menghapus tombol-tombol sebelumnya jika ada
    for widget in root.winfo_children():
        widget.destroy()
    
    # Judul di atas logo
    title_label = tk.Label(root, text="Selamat datang di program analisis sentiment menggunakan Naive Bayes", font=("Helvetica", 16), bg='lightblue')
    title_label.pack()

    # Logo Universitas
    show_logo("university_logo.png", root)

    # Nama dan NIM Mahasiswa
    student_info_label = tk.Label(root, text="Nama: Rizky Yugo Pratama\nNIM: 1911510343", bg='lightblue')
    student_info_label.pack()

    # Menampilkan dialog import CSV
    df_raw = import_csv()

    if df_raw is not None:
        # Tombol untuk menampilkan data mentah
        show_raw_data_button = tk.Button(root, text="Show Raw Data", command=lambda: show_raw_data(df_raw), bg='skyblue', fg='white', font=('Arial', 12))
        show_raw_data_button.pack()

        # Pra-pemrosesan data
        df_processed = preprocess_dataset(df_raw)

        # Menampilkan data yang telah diproses dalam bentuk tabel
        show_preprocessed_data_button = tk.Button(root, text="Show Preprocessed Data", command=lambda: show_preprocessed_data(df_processed), bg='skyblue', fg='white', font=('Arial', 12))
        show_preprocessed_data_button.pack()

        # Tombol untuk menyimpan data yang telah diproses
        save_preprocessed_data_button = tk.Button(root, text="Save Preprocessed Data", command=lambda: save_preprocessed_data(df_processed), bg='skyblue', fg='white', font=('Arial', 12))
        save_preprocessed_data_button.pack()

        # Tombol untuk melakukan proses preprocessing lagi
        preprocessing_again_button = tk.Button(root, text="Preprocess Again", command=preprocessing_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        preprocessing_again_button.pack()

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Preprocessing and Dataset Display")
root.configure(bg='lightblue')  # Mengubah warna background menjadi biru muda

# Judul di atas logo
title_label = tk.Label(root, text="Selamat Datang di Program Preprosessing", font=("Helvetica", 16), bg='lightblue')
title_label.pack()

# Logo Universitas
show_logo("university_logo.png", root)

# Nama dan NIM Mahasiswa
student_info_label = tk.Label(root, text="Nama: Rizky Yugo Pratama\nNIM: 1911510343", bg='lightblue')
student_info_label.pack()

# Tombol Preprocessing
preprocessing_button = tk.Button(root, text="Preprocessing", command=preprocessing_button_click, bg='skyblue', fg='white', font=('Arial', 12))
preprocessing_button.pack()

# Menjalankan main loop GUI
root.mainloop()
