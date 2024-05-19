import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fungsi untuk membaca file CSV dan melakukan prediksi sentimen pada data uji
def predict_sentiment():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
        except UnicodeDecodeError:
            # Jika gagal membaca dengan encoding utf-8, coba menggunakan encoding lain
            data = pd.read_csv(file_path, encoding='latin-1')
        
        # Preprocess data uji (pastikan praproses sama dengan data latih)
        X_test = data['Processed_Text']
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        # Prediksi sentimen pada data uji
        y_pred = naive_bayes.predict(X_test_tfidf)
        
        # Menampilkan hasil prediksi dalam tabel di GUI
        data['Predicted_Sentiment'] = y_pred
        show_prediction(data)
        
        # Menampilkan grafik perbandingan sentimen negatif, positif, dan netral
        show_sentiment_graph(data)

# Fungsi untuk menampilkan hasil prediksi dalam tabel di GUI
def show_prediction(data):
    # Membuat jendela baru untuk menampilkan tabel hasil prediksi
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Prediction Results")
    
    # Membuat tabel
    tree = ttk.Treeview(prediction_window)
    tree["columns"] = list(data.columns)
    tree["show"] = "headings"
    for column in data.columns:
        tree.heading(column, text=column)
    for index, row in data.iterrows():
        # Menambahkan nilai "Netral" jika hasil prediksi adalah 0
        if row['Predicted_Sentiment'] == 0:
            tree.insert("", tk.END, values=list(row) + ['Netral'])
        else:
            tree.insert("", tk.END, values=list(row))
    tree.pack(expand=True, fill="both")
    
    # Tombol untuk menyimpan hasil prediksi ke file CSV
    save_button = tk.Button(prediction_window, text="Save Prediction", command=lambda: save_prediction(data))
    save_button.pack()


def show_sentiment_graph(data):
    # Menghitung jumlah sentimen negatif, positif, dan netral
    sentiment_counts = data['Predicted_Sentiment'].value_counts()
    
    # Menambahkan nilai 0 jika tidak ada sentimen netral dalam data
    if 0 not in sentiment_counts.index:
        sentiment_counts[0] = 0
    
    # Membuat plot
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    
    # Menambahkan label pada sumbu x
    ax.set_xticks(range(len(sentiment_counts)))  # Menyesuaikan jumlah lokasi tetap dengan jumlah label
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    
    # Menampilkan plot di GUI
    graph_frame = tk.Frame(root)
    graph_frame.pack()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Fungsi untuk menyimpan hasil prediksi ke file CSV
def save_prediction(data):
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path:
        data.to_csv(save_path, index=False)
        messagebox.showinfo("Information", "Prediction saved successfully!")

# Fungsi untuk menampilkan logo (jika diperlukan)
def show_logo(logo_path, root):
    logo = tk.PhotoImage(file=logo_path)
    logo_label = tk.Label(root, image=logo, bg='lightblue')
    logo_label.image = logo  # Menjaga referensi agar gambar tidak terhapus
    logo_label.pack()

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Sentiment Prediction using Naive Bayes")
root.configure(bg='lightblue')  # Mengubah warna background menjadi biru muda

# Judul di atas logo
title_label = tk.Label(root, text="Selamat Datang Di Prediksi Sentiment Menggunaan Naive Bayes", font=("Helvetica", 16), bg='lightblue')
title_label.pack()

# Logo Universitas
show_logo("university_logo.png", root)  # Ganti dengan path logo yang benar

# Nama dan NIM Mahasiswa
student_info_label = tk.Label(root, text="Name: Your Name\nNIM: Your NIM", bg='lightblue')
student_info_label.pack()

# Tombol untuk melakukan prediksi sentimen pada data uji
predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment, bg='skyblue', fg='white', font=('Arial', 12))
predict_button.pack()

# Load trained model and TF-IDF vectorizer
naive_bayes = joblib.load('naive_bayes_model.pkl')  # Load trained Naive Bayes model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load TF-IDF vectorizer

# Menjalankan main loop GUI
root.mainloop()
