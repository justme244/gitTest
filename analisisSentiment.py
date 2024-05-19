import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Fungsi untuk membaca file CSV dan melakukan analisis sentimen
def analyze_sentiment():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        data = None
        for enc in encodings:
            try:
                # Membaca file CSV dengan pengaturan delimiter
                data = pd.read_csv(file_path, encoding=enc, delimiter=';')
                print(f"File opened successfully with encoding: {enc}")
                break
            except Exception as e:
                last_exception = e

        if data is None:
            messagebox.showerror("Error", f"Failed to open file: {last_exception}")
            return

        # Tampilkan nama kolom untuk debug
        print("Columns in the CSV file:", data.columns.tolist())
        messagebox.showinfo("Debug Info", f"Columns in the CSV file: {data.columns.tolist()}")

        if 'Processed_Text' not in data.columns or 'Sentiment' not in data.columns:
            messagebox.showerror("Error", "CSV file must contain 'Processed_Text' and 'Sentiment' columns.")
            return

        # Deteksi dan penanganan nilai NaN
        if data.isnull().values.any():
            messagebox.showwarning("Warning", "Data contains NaN values. Preprocessing...")
            data.dropna(inplace=True)  # Menghapus baris yang mengandung NaN
        
        X = data['Processed_Text']
        y = data['Sentiment']
        
        # Pembagian data menjadi data latih dan data uji (misalnya: 80% latih, 20% uji)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ekstraksi fitur menggunakan TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        
        # Pelatihan model Naive Bayes
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_tfidf, y_train)
        
        # Simpan model Naive Bayes
        joblib.dump(naive_bayes, 'naive_bayes_model.pkl')
        
        # Simpan model TfidfVectorizer
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        
        # Evaluasi model
        evaluate_sentiment(X_test, y_test, tfidf_vectorizer, naive_bayes)
        
        messagebox.showinfo("Information", "Sentiment analysis completed successfully!")

# Fungsi untuk menampilkan logo (jika diperlukan)
def show_logo(logo_path, root):
    logo = tk.PhotoImage(file=logo_path)
    logo_label = tk.Label(root, image=logo, bg='lightblue')
    logo_label.image = logo  # Menjaga referensi agar gambar tidak terhapus
    logo_label.pack()

# Fungsi untuk button click preprocessing (jika diperlukan)
def preprocessing_button_click():
    # Di sini Anda dapat menambahkan kode untuk button click preprocessing
    pass

# Fungsi untuk menampilkan evaluasi model
def evaluate_sentiment(X_test, y_test, tfidf_vectorizer, naive_bayes):
    # Ekstraksi fitur menggunakan TF-IDF
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Prediksi sentimen pada data uji
    y_pred = naive_bayes.predict(X_test_tfidf)
    
    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Menampilkan hasil evaluasi
    messagebox.showinfo("Evaluation Results", 
                        f"Accuracy: {accuracy:.2f}\n"
                        f"Precision: {precision:.2f}\n"
                        f"Recall: {recall:.2f}\n"
                        f"F1 Score: {f1:.2f}")

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Sentiment Analysis using Naive Bayes")
root.configure(bg='lightblue')  # Mengubah warna background menjadi biru muda

# Judul di atas logo
title_label = tk.Label(root, text="Selamat datang di program analisis sentiment menggunakan Naive Bayes", font=("Helvetica", 16), bg='lightblue')
title_label.pack()

# Logo Universitas
show_logo("university_logo.png", root)  # Ganti dengan path logo yang benar

# Nama dan NIM Mahasiswa
student_info_label = tk.Label(root, text="Nama: Rizky Yugo Pratama\nNIM: 1911510343", bg='lightblue')
student_info_label.pack()

# Tombol untuk melakukan analisis sentimen
analyze_button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment, bg='skyblue', fg='white', font=('Arial', 12))
analyze_button.pack()

# Menjalankan main loop GUI
root.mainloop()
