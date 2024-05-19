import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd

# Placeholder function for saving labels
def save_labels():
    # Mengambil data dari setiap baris dan menyimpannya ke dalam DataFrame
    labeled_data = []
    for item_id in tree.get_children():
        index = tree.item(item_id, 'values')[0]
        text = tree.item(item_id, 'values')[1]
        sentiment = tree.item(item_id, 'values')[2]
        labeled_data.append({'Index': index, 'Processed_Text': text, 'Sentiment': sentiment})
    
    # Mengonversi ke DataFrame
    labeled_df = pd.DataFrame(labeled_data)
    
    # Menyimpan DataFrame ke file CSV
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path:
        labeled_df.to_csv(save_path, index=False)
        messagebox.showinfo("Info", "Labels saved successfully!")

# Fungsi untuk membuka file CSV dan menampilkan isinya pada GUI
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                display_data(df)
                break
            except Exception as e:
                last_exception = e
        else:
            messagebox.showerror("Error", f"Failed to open file: {last_exception}")

# Fungsi untuk menampilkan data pada GUI
def display_data(df):
    tree.delete(*tree.get_children())
    for index, row in df.iterrows():
        tree.insert("", "end", values=(index, row["Processed_Text"], ""))

# Fungsi untuk melakukan pelabelan manual
def label_data():
    selected_items = tree.selection()
    if len(selected_items) == 0:
        messagebox.showwarning("Warning", "Please select a row to label.")
        return
    for item in selected_items:
        if selected_sentiment.get() == "":
            messagebox.showwarning("Warning", "Please select a sentiment.")
            return
        tree.item(item, values=(tree.item(item)["values"][0], tree.item(item)["values"][1], selected_sentiment.get()))
        selected_sentiment.set("")  # Mengatur kembali nilai radio button setelah digunakan

# Fungsi untuk menangani pemilihan radio button
def handle_radio_button(event):
    selected_sentiment.set(event.widget["value"])

# Membuat GUI dengan Tkinter
root = tk.Tk()
root.title("Preprocessing and Dataset Display")
root.configure(bg='lightblue')

# Frame untuk tombol-tombol
button_frame = tk.Frame(root, bg="lightblue")
button_frame.pack(pady=10)

# Tombol untuk membuka file
open_button = tk.Button(button_frame, text="Open File", command=open_file)
open_button.grid(row=0, column=0, padx=5)

# Tombol untuk menyimpan label
save_button = tk.Button(button_frame, text="Save Labels", command=save_labels)
save_button.grid(row=0, column=1, padx=5)

# Tombol untuk melakukan pelabelan manual
label_button = tk.Button(button_frame, text="Label Data", command=label_data)
label_button.grid(row=0, column=2, padx=5)

# Radio buttons untuk memilih sentimen
sentiment_frame = tk.LabelFrame(root, text="Sentiment", bg='lightblue')
sentiment_frame.pack(pady=5)

selected_sentiment = tk.StringVar()  # Variabel untuk menyimpan pilihan sentimen

positif_radio = tk.Radiobutton(sentiment_frame, text="Positif", variable=selected_sentiment, value="Positif", bg='lightblue')
positif_radio.grid(row=0, column=0, padx=5)
positif_radio.bind("<Button-1>", handle_radio_button)

netral_radio = tk.Radiobutton(sentiment_frame, text="Netral", variable=selected_sentiment, value="Netral", bg='lightblue')
netral_radio.grid(row=0, column=1, padx=5)
netral_radio.bind("<Button-1>", handle_radio_button)

negatif_radio = tk.Radiobutton(sentiment_frame, text="Negatif", variable=selected_sentiment, value="Negatif", bg='lightblue')
negatif_radio.grid(row=0, column=2, padx=5)
negatif_radio.bind("<Button-1>", handle_radio_button)

# Tabel untuk menampilkan data
tree_frame = tk.Frame(root)
tree_frame.pack(padx=10, pady=5)

tree = ttk.Treeview(tree_frame, columns=("Index", "Processed_Text", "Sentiment"), show='headings')
tree.heading("Index", text="Index")
tree.heading("Processed_Text", text="Processed_Text")
tree.heading("Sentiment", text="Sentiment")
tree.pack()

# Menjalankan main loop GUI
root.mainloop()
