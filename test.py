import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import pickle 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Download stopwords if not already downloaded
download('stopwords')

class SentimentAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis using Naive Bayes")
        self.configure(bg='lightblue')

        # Title above logo
        self.title_label = tk.Label(self, text="Selamat datang di program analisis sentiment menggunakan Naive Bayes", font=("Helvetica", 16), bg='lightblue')
        self.title_label.pack()

        # University Logo
        self.show_logo("university_logo.png")

        # Student Info
        self.student_info_label = tk.Label(self, text="Nama: Rizky Yugo Pratama\nNIM: 1911510343", bg='lightblue')
        self.student_info_label.pack()

        # Main Buttons
        self.preprocessing_button = tk.Button(self, text="Preprocessing", command=self.preprocessing_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.preprocessing_button.pack(pady=5)

        self.labeling_button = tk.Button(self, text="Labeling", command=self.labeling_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.labeling_button.pack(pady=5)

        self.analyze_button = tk.Button(self, text="Analyze Sentiment", command=self.analyze_sentiment_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.analyze_button.pack(pady=5)

        self.test_button = tk.Button(self, text="Test Model", command=self.test_model_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.test_button.pack(pady=5)

    def show_logo(self, image_path):
        img = tk.PhotoImage(file=image_path)
        label = tk.Label(self, image=img, bg='lightblue')
        label.image = img  # Keep a reference to avoid garbage collection
        label.pack()

    def preprocessing_button_click(self):
        self.clear_widgets()
        self.title_label.pack()
        self.show_logo("university_logo.png")
        self.student_info_label.pack()

        df_raw = self.import_csv()

        if df_raw is not None:
            self.show_raw_data_button = tk.Button(self, text="Show Raw Data", command=lambda: self.show_raw_data(df_raw), bg='skyblue', fg='white', font=('Arial', 12))
            self.show_raw_data_button.pack()

            df_processed = self.preprocess_dataset(df_raw)

            self.show_preprocessed_data_button = tk.Button(self, text="Show Preprocessed Data", command=lambda: self.show_preprocessed_data(df_processed), bg='skyblue', fg='white', font=('Arial', 12))
            self.show_preprocessed_data_button.pack()

            self.save_preprocessed_data_button = tk.Button(self, text="Save Preprocessed Data", command=lambda: self.save_preprocessed_data(df_processed), bg='skyblue', fg='white', font=('Arial', 12))
            self.save_preprocessed_data_button.pack()

            self.preprocessing_again_button = tk.Button(self, text="Preprocess Again", command=self.preprocessing_button_click, bg='skyblue', fg='white', font=('Arial', 12))
            self.preprocessing_again_button.pack()

        self.back_button = tk.Button(self, text="Back to Home", command=self.back_to_home, bg='skyblue', fg='white', font=('Arial', 12))
        self.back_button.pack(pady=5)

    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                messagebox.showinfo("Info", f"{len(df)} baris telah diimpor dari file CSV.")
                return df
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        return None

    def preprocess_text(self, text):
        text = re.sub(r"(http[s]?\://\S+)|([\[\(].*[\)\]])|([#@]\S+)|\n", "", text)
        text = re.sub('\s+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()

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

        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word not in stop_words]

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

    def preprocess_dataset(self, df):
        df['Processed_Text'] = df['full_text'].apply(self.preprocess_text)
        df_processed = df[['Processed_Text']].copy()
        return df_processed

    def save_preprocessed_data(self, df):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Info", "Data berhasil disimpan ke dalam file CSV.")
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan saat menyimpan data: {str(e)}")

    def show_raw_data(self, df):
        self.show_data(df, "Raw Data")

    def show_preprocessed_data(self, df):
        self.show_data(df, "Preprocessed Data")

    def show_data(self, df, title):
        top = tk.Toplevel(self)
        top.title(title)
        top.geometry("800x400")

        frame = tk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        for column in df.columns:
            tree.heading(column, text=column)
        for index, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

    def labeling_button_click(self):
        self.clear_widgets()
        self.title_label.pack()
        self.show_logo("university_logo.png")
        self.student_info_label.pack()

        self.open_button = tk.Button(self, text="Open File", command=self.open_file, bg='skyblue', fg='white', font=('Arial', 12))
        self.open_button.pack(pady=5)

        self.save_button = tk.Button(self, text="Save Labels", command=self.save_labels, bg='skyblue', fg='white', font=('Arial', 12))
        self.save_button.pack(pady=5)

        self.label_button = tk.Button(self, text="Label Data", command=self.label_data, bg='skyblue', fg='white', font=('Arial', 12))
        self.label_button.pack(pady=5)

        self.sentiment_frame = tk.LabelFrame(self, text="Sentiment", bg='lightblue')
        self.sentiment_frame.pack(pady=5)

        self.selected_sentiment = tk.StringVar()

        self.positif_radio = tk.Radiobutton(self.sentiment_frame, text="Positif", variable=self.selected_sentiment, value="Positif", bg='lightblue', font=('Arial', 12))
        self.positif_radio.pack(side=tk.LEFT, padx=5)

        self.negatif_radio = tk.Radiobutton(self.sentiment_frame, text="Negatif", variable=self.selected_sentiment, value="Negatif", bg='lightblue', font=('Arial', 12))
        self.negatif_radio.pack(side=tk.LEFT, padx=5)

        self.netral_radio = tk.Radiobutton(self.sentiment_frame, text="Netral", variable=self.selected_sentiment, value="Netral", bg='lightblue', font=('Arial', 12))
        self.netral_radio.pack(side=tk.LEFT, padx=5)

        self.label_list = []
        self.file_data = None

        self.back_button = tk.Button(self, text="Back to Home", command=self.back_to_home, bg='skyblue', fg='white', font=('Arial', 12))
        self.back_button.pack(pady=5)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.file_data = pd.read_csv(file_path)
                self.label_list = [None] * len(self.file_data)
                messagebox.showinfo("Info", f"{len(self.file_data)} baris telah diimpor dari file CSV.")
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

    def label_data(self):
        if self.file_data is not None:
            top = tk.Toplevel(self)
            top.title("Label Data")
            top.geometry("800x600")

            self.current_index = 0

            self.text_label = tk.Label(top, text="", wraplength=700, justify=tk.LEFT, bg='lightblue', font=('Arial', 12))
            self.text_label.pack(pady=10)

            self.prev_button = tk.Button(top, text="Previous", command=self.show_previous, bg='skyblue', fg='white', font=('Arial', 12))
            self.prev_button.pack(side=tk.LEFT, padx=10, pady=5)

            self.next_button = tk.Button(top, text="Next", command=self.show_next, bg='skyblue', fg='white', font=('Arial', 12))
            self.next_button.pack(side=tk.RIGHT, padx=10, pady=5)

            self.update_text_label()

    def update_text_label(self):
        if self.file_data is not None and 0 <= self.current_index < len(self.file_data):
            text = self.file_data.iloc[self.current_index]['Processed_Text']
            self.text_label.config(text=text)
            if self.label_list[self.current_index] is not None:
                self.selected_sentiment.set(self.label_list[self.current_index])
            else:
                self.selected_sentiment.set("")

    def show_previous(self):
        self.save_current_label()
        if self.current_index > 0:
            self.current_index -= 1
            self.update_text_label()

    def show_next(self):
        self.save_current_label()
        if self.current_index < len(self.file_data) - 1:
            self.current_index += 1
            self.update_text_label()

    def save_current_label(self):
        sentiment = self.selected_sentiment.get()
        if sentiment:
            self.label_list[self.current_index] = sentiment

    def save_labels(self):
        if self.file_data is not None and self.label_list:
            self.file_data['Sentiment'] = self.label_list
            self.save_preprocessed_data(self.file_data)

    def analyze_sentiment_button_click(self):
        self.clear_widgets()
        self.title_label.pack()
        self.show_logo("university_logo.png")
        self.student_info_label.pack()

        self.file_path_entry = tk.Entry(self, width=50)
        self.file_path_entry.pack(pady=5)

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_file, bg='skyblue', fg='white', font=('Arial', 12))
        self.browse_button.pack(pady=5)

        self.model = None
        self.vectorizer = None

        self.train_model_button = tk.Button(self, text="Train Model", command=self.train_model_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.train_model_button.pack(pady=5)

        self.analyze_sentiment_button = tk.Button(self, text="Analyze Sentiment", command=self.analyze_sentiment_button_click_confirm, bg='skyblue', fg='white', font=('Arial', 12))
        self.analyze_sentiment_button.pack(pady=5)

        self.back_button = tk.Button(self, text="Back to Home", command=self.back_to_home, bg='skyblue', fg='white', font=('Arial', 12))
        self.back_button.pack(pady=5)

    def analyze_sentiment_button_click_confirm(self):
        file_path = self.file_path_entry.get()
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if 'Processed_Text' not in df.columns:
                    raise ValueError("CSV file does not contain 'Processed_Text' column.")
                self.analyze_sentiment(df)
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

    def train_model_button_click(self):
        file_path = self.file_path_entry.get()
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if 'Processed_Text' not in df.columns or 'Sentiment' not in df.columns:
                    raise ValueError("CSV file does not contain 'Processed_Text' or 'Sentiment' column.")
                self.train_model(df)
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

    def analyze_sentiment(self, df):
        if self.model is None or self.vectorizer is None:
            messagebox.showerror("Error", "Model belum dilatih. Silakan latih model terlebih dahulu.")
            return

        df['Sentiment_Prediction'] = self.model.predict(self.vectorizer.transform(df['Processed_Text']))

        self.show_analysis_result(df)

    def show_analysis_result(self, df):
        top = tk.Toplevel(self)
        top.title("Sentiment Analysis Result")
        top.geometry("800x600")

        frame = tk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        for column in df.columns:
            tree.heading(column, text=column)
        for index, row in df.iterrows():
            tree.insert("", tk.END, values=list(row))
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

    def train_model(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['Processed_Text'], df['Sentiment'], test_size=0.2, random_state=42)

        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.model = MultinomialNB()
        self.model.fit(X_train_tfidf, y_train)

        y_pred = self.model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        self.show_evaluation_result(accuracy, precision, recall, f1)

    def show_evaluation_result(self, accuracy, precision, recall, f1):
        top = tk.Toplevel(self)
        top.title("Evaluation Result")
        top.geometry("400x300")

        result_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}"
        result_label = tk.Label(top, text=result_text, font=('Arial', 12), justify=tk.LEFT)
        result_label.pack(pady=20)

    def test_model_button_click(self):
        self.clear_widgets()
        self.title_label.pack()
        self.show_logo("university_logo.png")
        self.student_info_label.pack()

        self.model_path_entry = tk.Entry(self, width=50)
        self.model_path_entry.pack(pady=5)

        self.browse_model_button = tk.Button(self, text="Browse Model", command=self.browse_model_file, bg='skyblue', fg='white', font=('Arial', 12))
        self.browse_model_button.pack(pady=5)

        self.vectorizer_path_entry = tk.Entry(self, width=50)
        self.vectorizer_path_entry.pack(pady=5)

        self.browse_vectorizer_button = tk.Button(self, text="Browse Vectorizer", command=self.browse_vectorizer_file, bg='skyblue', fg='white', font=('Arial', 12))
        self.browse_vectorizer_button.pack(pady=5)

        self.test_button = tk.Button(self, text="Test Model", command=self.test_model, bg='skyblue', fg='white', font=('Arial', 12))
        self.test_button.pack(pady=5)

        self.back_button = tk.Button(self, text="Back to Home", command=self.back_to_home, bg='skyblue', fg='white', font=('Arial', 12))
        self.back_button.pack(pady=5)

    def test_model(self):
        model_path = self.model_path_entry.get()
        vectorizer_path = self.vectorizer_path_entry.get()
        if model_path and vectorizer_path:
            try:
                with open(model_path, 'rb') as model_file:
                    self.model = pickle.load(model_file)
                with open(vectorizer_path, 'rb') as vectorizer_file:
                    self.vectorizer = pickle.load(vectorizer_file)
                messagebox.showinfo("Info", "Model dan Vectorizer berhasil dimuat.")
            except Exception as e:
                messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def browse_model_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, file_path)

    def browse_vectorizer_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            self.vectorizer_path_entry.delete(0, tk.END)
            self.vectorizer_path_entry.insert(0, file_path)

    def save_preprocessed_data(self, df):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Info", f"Data berhasil disimpan di {file_path}")

    def back_to_home(self):
        self.clear_widgets()
        self.title_label.pack(pady=20)
        self.show_logo("university_logo.png")
        self.student_info_label.pack(pady=10)

        self.preprocessing_button = tk.Button(self, text="Preprocess Data", command=self.preprocess_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.preprocessing_button.pack(pady=10)

        self.labeling_button = tk.Button(self, text="Label Data", command=self.labeling_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.labeling_button.pack(pady=10)

        self.analyze_sentiment_button = tk.Button(self, text="Analyze Sentiment", command=self.analyze_sentiment_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.analyze_sentiment_button.pack(pady=10)

        self.test_model_button = tk.Button(self, text="Test Model", command=self.test_model_button_click, bg='skyblue', fg='white', font=('Arial', 12))
        self.test_model_button.pack(pady=10)

    def clear_widgets(self):
        for widget in self.winfo_children():
            widget.pack_forget()

if __name__ == "__main__":
    app = SentimentAnalysisApp()
    app.mainloop()
