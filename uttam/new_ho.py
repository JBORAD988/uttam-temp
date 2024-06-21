import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Function to load CSV data
def load_csv(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None


# Function to evaluate algorithms
def evaluate_algorithms(X, y):
    # Encode target if it is categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(random_state=42)
    }

    # Fit and predict using each classifier
    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    # Determine which algorithm has the best accuracy
    best_algorithm = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_algorithm]
    return best_algorithm, best_accuracy, accuracies


# Main function
def main():
    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            df = load_csv(file_path)
            if df is None:
                messagebox.showerror("Error", f"{file_path} not found.")
                return
            display_columns(df)

    def display_columns(df):
        global data_frame
        data_frame = df

        for widget in selection_frame.winfo_children():
            widget.destroy()

        # Checkbuttons for selecting columns
        global column_var
        column_var = []
        for column in df.columns[:-1]:  # Exclude target column
            var = tk.IntVar()
            column_var.append(var)
            check_button = ttk.Checkbutton(selection_frame, text=column, variable=var)
            check_button.pack(anchor=tk.W)

        # Button to confirm column selection
        confirm_button = ttk.Button(selection_frame, text="Confirm Selection", command=lambda: select_columns(df))
        confirm_button.pack(pady=10)

    def select_columns(df):
        selected_columns = [df.columns[i] for i, var in enumerate(column_var) if var.get() == 1]

        if not selected_columns:
            messagebox.showerror("Error", "Please select at least one feature column.")
            return

        for widget in input_frame.winfo_children():
            widget.destroy()

        user_inputs.clear()
        for column in selected_columns:
            label = ttk.Label(input_frame, text=f"Enter data for '{column}' column:")
            label.pack()

            user_input = ttk.Entry(input_frame, width=20)
            user_input.pack()
            user_inputs[column] = user_input

        submit_button = ttk.Button(input_frame, text="Submit", command=lambda: evaluate(df, selected_columns))
        submit_button.pack(pady=10)

    def evaluate(df, selected_columns):
        user_data = {}
        for column, entry in user_inputs.items():
            user_data[column] = entry.get().strip()

        if not all(user_data.values()):
            messagebox.showerror("Error", "Please enter data for all selected columns.")
            return

        # Prepare user data as DataFrame
        user_df = pd.DataFrame([user_data])

        # Handle categorical data
        for column in selected_columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                le.fit(df[column].astype(str))
                user_df[column] = le.transform(user_df[column])

        # Prepare full data for evaluation
        features = df[selected_columns]
        target_column = df.columns[-1]
        target = df[target_column]

        # Encode categorical features in the DataFrame
        for column in selected_columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                features[column] = le.fit_transform(features[column].astype(str))

        best_algorithm, best_accuracy, accuracies = evaluate_algorithms(features, target)

        result_text = f"The best algorithm is '{best_algorithm}' with accuracy: {best_accuracy:.2f}\n\n"
        result_text += "All algorithm accuracies:\n"
        for algo, acc in accuracies.items():
            result_text += f"{algo}: {acc:.2f}\n"

        messagebox.showinfo("Algorithm Evaluation Results", result_text)

    # Initialize global variables
    global user_inputs, data_frame
    user_inputs = {}
    data_frame = None

    # Create the main window
    root = tk.Tk()
    root.title("Feature Selection and Algorithm Evaluation")

    # Frames for column selection and user input
    file_frame = ttk.Frame(root)
    file_frame.pack(padx=10, pady=10)

    selection_frame = ttk.LabelFrame(root, text="Select Columns for Input")
    selection_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    input_frame = ttk.LabelFrame(root, text="Enter Data for Selected Columns")
    input_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Button to open CSV file
    open_button = ttk.Button(file_frame, text="Open CSV File", command=open_file)
    open_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
