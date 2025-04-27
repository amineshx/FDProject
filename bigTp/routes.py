from flask import render_template, request, redirect, url_for, flash, session
from bigTp import app
import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

UPLOAD_FOLDER = 'bigTp/uploads'  # We'll create this folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Your load_dataset function
def load_dataset(file_path, normalize=True):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.trff'):
        df = pd.read_csv(file_path, delimiter='\t')
    else:
        raise ValueError("Format de fichier non supporté. Utilisez .csv ou .trff")
    
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_numeric.values)
    else:
        data = df_numeric.values

    return df, data

def plot_elbow_curve(data, max_k=10):
    """
    Trace la courbe d'Elbow pour déterminer le nombre optimal de clusters.
    """
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='-', color='blue')
    plt.title("Méthode du Coude (Elbow Method)")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie (Within-Cluster Sum of Squares)")
    plt.grid(True)

    # Save the figure to the correct static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    elbow_path = os.path.join(static_dir, 'elbow.png')
    plt.savefig(elbow_path)
    plt.close()
    print(f"Saving elbow.png to: {elbow_path}")
    return elbow_path

def explore_dataset(df):
    """
    Affiche un résumé utile de la structure d'un DataFrame :
    - les premières lignes
    - la forme
    - les types de données
    - les valeurs manquantes
    - les statistiques descriptives
    
    Paramètre :
    - df : le DataFrame à explorer
    """
    print("🔹 Aperçu des 5 premières lignes :")
    print(df.head())
    print("\n🔹 Dimensions du dataset :", df.shape)
    print("\n🔹 Types de données :")
    print(df.dtypes)
    print("\n🔹 Valeurs manquantes par colonne :")
    print(df.isnull().sum())
    print("\n🔹 Statistiques descriptives (colonnes numériques) :")
    print(df.describe())

def plot_boxplots(data, columns=None):
    """
    Affiche les boxplots pour chaque colonne du jeu de données normalisé.

    Paramètres :
    - data : données normalisées (numpy array ou DataFrame)
    - columns : noms des colonnes (si data est un numpy array)
    """
    # Si data est un array NumPy, le convertir en DataFrame pour tracer
    if not isinstance(data, pd.DataFrame):
        if columns is None:
            columns = [f"Var{i+1}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=columns)

    # Tracer les boxplots
    plt.figure(figsize=(12, 6))
    data.boxplot()
    plt.title("Boxplots des variables (normalisées)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure to the static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    boxplot_path = os.path.join(static_dir, 'boxplots.png')
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saving boxplots.png to: {boxplot_path}")
    return boxplot_path

def select_and_plot_scatter(data, col1, col2):
    """
    Trace un scatter plot entre deux colonnes choisies.

    Paramètres :
    - data : DataFrame
    - col1 : nom de la première colonne
    - col2 : nom de la deuxième colonne
    """
    # Tracer le scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data[col1], data[col2], c='blue', edgecolor='k')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"Scatter Plot entre {col1} et {col2}")
    plt.grid(True)

    # Save the figure to the static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    scatter_path = os.path.join(static_dir, 'scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saving scatter.png to: {scatter_path}")
    return scatter_path


@app.route('/generate_scatter', methods=['POST'])
def generate_scatter():
    try:
        col1 = request.form['col1']
        col2 = request.form['col2']

        # Reload the dataset from the session
        filepath = session.get('uploaded_file')
        if not filepath:
            return "No dataset found. Please upload a dataset first.", 400

        df, _ = load_dataset(filepath, normalize=False)

        # Generate scatter plot
        scatter_image = select_and_plot_scatter(df, col1, col2)

        # Return only the scatter plot section as HTML
        return render_template('scatter_plot.html', scatter_image='scatter.png', time=time.time)
    except Exception as e:
        return str(e), 500
    
@app.route('/', methods=['GET', 'POST'])
def home():
    scatter_image = None
    columns = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            
            if file:
                # Save the file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                try:
                    # Store the file path in the session
                    session['uploaded_file'] = filepath

                    df, data = load_dataset(filepath, normalize=True)
                    
                    # Generate dataset summary
                    dimensions = f"Dimensions: {df.shape}"
                    data_types = df.dtypes.to_frame(name="Data Type").reset_index().rename(columns={"index": "Column"}).to_html(classes='table table-striped', index=False)
                    missing_values = df.isnull().sum().to_frame(name="Missing Values").reset_index().rename(columns={"index": "Column"}).to_html(classes='table table-striped', index=False)
                    descriptive_stats = df.describe().to_html(classes='table table-striped')

                    dataset_summary = {
                        "dimensions": dimensions,
                        "data_types": data_types,
                        "missing_values": missing_values,
                        "descriptive_stats": descriptive_stats,
                    }

                    # Preview first 5 rows
                    preview = df.head().to_html(classes='table table-striped', border=0)

                    # Generate elbow curve
                    plot_elbow_curve(data)

                    # Generate boxplots
                    plot_boxplots(data, columns=df.select_dtypes(include=['float64', 'int64']).columns)

                    # Pass filenames for the images
                    elbow_image = 'elbow.png'
                    boxplot_image = 'boxplots.png'

                    # Pass column names for scatter plot selection
                    columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

                    return render_template('home.html', preview=preview, elbow_image=elbow_image, boxplot_image=boxplot_image, dataset_summary=dataset_summary, columns=columns)
                except Exception as e:
                    flash(str(e))
                    return redirect(request.url)

        # Handle scatter plot generation
        elif 'col1' in request.form and 'col2' in request.form:
            col1 = request.form['col1']
            col2 = request.form['col2']

            try:
                # Reload the dataset from the session
                filepath = session.get('uploaded_file')
                if not filepath:
                    flash("No dataset found. Please upload a dataset first.")
                    return redirect(request.url)

                df, _ = load_dataset(filepath, normalize=False)

                # Generate scatter plot
                scatter_image = select_and_plot_scatter(df, col1, col2)
            except Exception as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('home.html', scatter_image=scatter_image, columns=columns)