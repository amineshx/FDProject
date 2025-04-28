from flask import render_template, request, redirect, flash, session,send_file
import io
from bigTp import app
import os
import time
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'


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

def evaluate_clustering_metrics(data, labels):
    """
    Calcule et affiche le coefficient de silhouette.

    Paramètres :
    - data : données normalisées (array ou DataFrame)
    - labels : labels de clustering (liste ou array)

    Retour :
    - silhouette : score entre -1 et 1
    """
    silhouette = silhouette_score(data, labels)
    print(f"🟩 Coefficient de Silhouette : {silhouette:.3f}")
    return silhouette

def compute_clustering_metrics(data, labels, display=True):
    """
    Calcule et affiche les principales métriques de qualité pour un clustering.

    Paramètres :
    - data : données normalisées (numpy array ou DataFrame)
    - labels : labels prédits (array)
    - display : bool, si True, affiche les résultats

    Retour :
    - metrics_dict : dictionnaire contenant les scores
    """
    metrics_dict = {}

    metrics_dict["Silhouette Score"] = silhouette_score(data, labels)
    metrics_dict["Davies-Bouldin Index"] = davies_bouldin_score(data, labels)
    metrics_dict["Calinski-Harabasz Index"] = calinski_harabasz_score(data, labels)

    if display:
        print("📊 Métriques de Clustering :")
        for key, value in metrics_dict.items():
            print(f"  - {key}: {value:.3f}")

    return metrics_dict

def apply_kmeans_show_classes(df, data, n_clusters, class_col_candidates=['Classe', 'Species', 'target']):
    """
    Applique K-Means sur les données normalisées, affiche la visualisation et ajoute la vraie classe si disponible.

    Paramètres :
    - df : DataFrame original (non normalisé, contient potentiellement la classe)
    - data : données normalisées (array ou DataFrame sans la classe)
    - n_clusters : nombre de clusters
    - class_col_candidates : liste des noms possibles de la colonne contenant les vraies classes

    Retour :
    - df_clustered : DataFrame contenant les données, les clusters, et la classe réelle (si trouvée)
    - metrics_dict : dictionnaire contenant les scores de clustering
    """
    # Convertir en DataFrame si data est un array
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=df.select_dtypes(include=['float64', 'int64']).columns)

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)

    # Création du DataFrame final
    df_clustered = data.copy()
    df_clustered["Cluster"] = labels

    # Chercher une colonne de classe dans le DataFrame original
    for col in class_col_candidates:
        if col in df.columns:
            df_clustered["Classe"] = df[col].values
            break  # on ajoute la première trouvée

    # Calculer les métriques de clustering
    metrics_dict = compute_clustering_metrics(data, labels, display=True)

    # Affichage : scatter plot sur les 2 premières dimensions
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', edgecolor='k', label='Clusters')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=200, marker='X', label='Centroïdes')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title(f"K-Means Clustering (k={n_clusters})")
    plt.legend()
    plt.grid(True)

    # Save the figure to the static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    kmeans_path = os.path.join(static_dir, 'kmeans.png')
    plt.savefig(kmeans_path)
    plt.close()
    print(f"Saving kmeans.png to: {kmeans_path}")

    return df_clustered, metrics_dict, 'kmeans.png'

def apply_agnes_and_plot_dendrogram(data, n_clusters=3, method='ward'):
    """
    Applique AGNES et affiche le dendrogramme avec une ligne de coupe.

    Paramètres :
    - data : numpy array normalisé
    - n_clusters : nombre de clusters à créer
    - method : méthode de linkage ('ward', 'single', 'complete', 'average')

    Retour :
    - labels : clusters assignés
    """
    # 1. Créer la matrice de liaison
    linked = linkage(data, method=method)

    # 2. Tracer le dendrogramme
    plt.figure(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    
    # 🔴 Ajouter la ligne de coupe correspondant au nombre de clusters
    max_d = linked[-(n_clusters - 1), 2]  # Hauteur où les derniers clusters fusionnent
    plt.axhline(y=max_d, color='red', linestyle='--', label=f'{n_clusters} clusters')
    plt.legend()

    plt.title(f"Dendrogramme AGNES (méthode = {method})")
    plt.xlabel("Index des points")
    plt.ylabel("Distance")
    plt.tight_layout()

    # Save the dendrogram to the static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    dendrogram_path = os.path.join(static_dir, 'agnes_dendrogram.png')
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Saving agnes_dendrogram.png to: {dendrogram_path}")

    # 3. Clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(data)

    return labels, 'agnes_dendrogram.png'

def diana_clustering(data, n_clusters=3):
    """
    Implémentation de l'algorithme DIANA (DIvisive ANAlysis Clustering).
    """
    n_samples = data.shape[0]
    
    # Initialiser avec tous les points dans un seul cluster
    current_labels = np.zeros(n_samples, dtype=int)
    
    # Liste pour suivre les clusters à diviser
    clusters_to_process = [np.arange(n_samples)]
    
    # Continuer jusqu'à obtenir le nombre souhaité de clusters
    next_label = 1
    
    while len(clusters_to_process) < n_clusters:
        # Choisir le cluster avec la plus grande dispersion
        max_dispersion = -1
        selected_cluster_idx = -1
        
        for i, cluster_indices in enumerate(clusters_to_process):
            if len(cluster_indices) <= 1:
                continue  # Ne pas diviser les clusters avec un seul point
                
            # Calculer la matrice de distance pour ce cluster
            cluster_data = data[cluster_indices]
            dist_matrix = squareform(pdist(cluster_data, metric='euclidean'))
            
            # Calculer la dispersion (somme des distances intra-cluster)
            dispersion = np.sum(dist_matrix) / 2  # Diviser par 2 car la matrice est symétrique
            
            if dispersion > max_dispersion:
                max_dispersion = dispersion
                selected_cluster_idx = i
        
        if selected_cluster_idx == -1:
            break  # Aucun cluster divisible trouvé
        
        # Diviser le cluster sélectionné
        cluster_to_split = clusters_to_process[selected_cluster_idx]
        cluster_data = data[cluster_to_split]
        
        if len(cluster_to_split) <= 2:
            # Si le cluster n'a que 2 points, simplement les séparer
            splitter_a = [cluster_to_split[0]]
            splitter_b = cluster_to_split[1:]
        else:
            # Trouver le point le plus éloigné du centre
            center = np.mean(cluster_data, axis=0)
            distances_to_center = np.linalg.norm(cluster_data - center, axis=1)
            splitter_idx = np.argmax(distances_to_center)
            
            # Initialiser les deux sous-clusters
            splitter_a = [cluster_to_split[splitter_idx]]
            splitter_b = [idx for i, idx in enumerate(cluster_to_split) if i != splitter_idx]
            
            # Réaffecter les points en fonction de leur similitude
            changes = True
            while changes and len(splitter_b) > 0:
                changes = False
                
                # Calculer les distances moyennes de chaque point au splitter_a
                subset_a = data[splitter_a]
                subset_b = data[splitter_b]
                
                # Pour chaque point dans splitter_b, vérifier s'il devrait être déplacé vers splitter_a
                for i, point_idx in enumerate(splitter_b[:]):
                    point = data[point_idx].reshape(1, -1)
                    
                    if len(splitter_a) == 0:
                        continue
                        
                    # Calculer la distance moyenne au groupe A
                    avg_dist_a = np.mean(np.linalg.norm(subset_a - point, axis=1))
                    
                    # Calculer la distance moyenne au reste du groupe B
                    remaining_b = np.delete(subset_b, i, axis=0)
                    if len(remaining_b) > 0:
                        avg_dist_b = np.mean(np.linalg.norm(remaining_b - point, axis=1))
                    else:
                        avg_dist_b = float('inf')  # Si B est vide, considérer comme distance infinie
                    
                    # Si le point est plus proche du groupe A, le déplacer
                    if avg_dist_a < avg_dist_b:
                        splitter_a.append(point_idx)
                        splitter_b.remove(point_idx)
                        subset_a = data[splitter_a]
                        subset_b = data[splitter_b] if len(splitter_b) > 0 else np.array([])
                        changes = True
                        break  # Recommencer le processus
        
        # Mettre à jour les labels
        current_labels[splitter_a] = next_label
        next_label += 1
        
        # Mettre à jour la liste des clusters
        clusters_to_process.pop(selected_cluster_idx)
        clusters_to_process.append(np.array(splitter_a))
        clusters_to_process.append(np.array(splitter_b))
    
    return current_labels

def visualize_diana_clustering(data, n_clusters=3):
    """
    Applique l'algorithme DIANA et visualise les clusters.
    
    Paramètres:
    - data: numpy array normalisé
    - n_clusters: nombre de clusters à créer
    
    Retourne:
    - labels: résultat du clustering
    """
    # Appliquer DIANA
    labels = diana_clustering(data, n_clusters)
    
    # Visualiser les clusters en 2D
    plot_clusters_2d(data, labels, title=f"DIANA Clustering avec {n_clusters} clusters")
    
    # Afficher la distribution des points par cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Distribution des points par cluster:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} points")
    
    return labels

def plot_clusters_2d(data, labels, title="Visualisation des clusters"):
    """
    Visualise les clusters en 2D après réduction dimensionnelle.
    """
    # Réduction dimensionnelle avec PCA si nécessaire
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_
        print(f"Variance expliquée par les 2 premières composantes: {sum(explained_var)*100:.2f}%")
    else:
        data_2d = data
    
    # Obtenir le nombre de clusters uniques
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Définir une palette de couleurs
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Tracer les points
    plt.figure(figsize=(10, 8))
    for i, color in zip(unique_labels, colors):
        cluster_points = data_2d[labels == i]  # Ensure this is used only for indexing
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=50, c=[color], 
            label=f'Cluster {i}'
        )
    
    plt.title(title, fontsize=15)
    plt.xlabel("Composante 1" if data.shape[1] > 2 else "Dimension 1")
    plt.ylabel("Composante 2" if data.shape[1] > 2 else "Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

@app.route('/apply_diana', methods=['POST'])
def apply_diana():
    try:
        # Get the number of clusters from the form
        n_clusters = int(request.form['n_clusters'])

        # Reload the dataset from the session
        filepath = session.get('uploaded_file')
        if not filepath:
            return "No dataset found. Please upload a dataset first.", 400

        df, data = load_dataset(filepath, normalize=True)

        # Apply DIANA clustering
        labels = diana_clustering(data, n_clusters)

        # Visualize the clustering
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
        os.makedirs(static_dir, exist_ok=True)
        diana_image_path = os.path.join(static_dir, 'diana.png')
        plot_clusters_2d(data, labels, title=f"DIANA Clustering with {n_clusters} Clusters")
        plt.savefig(diana_image_path)
        plt.close()

        # Return the partial template for DIANA results
        return render_template(
            'diana_partial.html',
            diana_image='diana.png',
            labels=dict(zip(*np.unique(labels, return_counts=True))),
            time=time.time()  # Pass the current timestamp
        )
    except Exception as e:
        return str(e), 500
    
@app.route('/apply_agnes', methods=['POST'])
def apply_agnes():
    try:
        # Get the number of clusters and linkage method from the form
        n_clusters = int(request.form['n_clusters'])
        method = request.form.get('method', 'ward')  # Default to 'ward'

        # Reload the dataset from the session
        filepath = session.get('uploaded_file')
        if not filepath:
            return "No dataset found. Please upload a dataset first.", 400

        df, data = load_dataset(filepath, normalize=True)

        # Apply AGNES and generate the dendrogram
        labels, dendrogram_image = apply_agnes_and_plot_dendrogram(data, n_clusters, method)

        # Return the partial template for AGNES results
        return render_template(
            'agnes_partial.html',
            dendrogram_image=dendrogram_image,
            labels=dict(zip(*np.unique(labels, return_counts=True))),
            time=time.time()  # Pass the current timestamp
        )
    except Exception as e:
        return str(e), 500
    
@app.route('/download_clustered_data', methods=['GET'])
def download_clustered_data():
    try:
        # Reload the dataset from the session
        filepath = session.get('uploaded_file')
        if not filepath:
            return "No dataset found. Please upload a dataset first.", 400

        # Load the dataset and normalize it
        df, data = load_dataset(filepath, normalize=True)

        # Apply K-Means clustering (default to 3 clusters for simplicity)
        n_clusters = session.get('n_clusters', 3)  # Use the last selected number of clusters
        df_clustered, _ = apply_kmeans_show_classes(df, data, n_clusters)

        # Save the clustered data to a CSV in memory
        output = io.StringIO()
        df_clustered.to_csv(output, index=False)
        output.seek(0)

        # Serve the CSV file as a download
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='clustered_data.csv'
        )
    except Exception as e:
        return str(e), 500


@app.route('/apply_kmeans', methods=['POST'])
def apply_kmeans():
    try:
        # Get the number of clusters from the form
        n_clusters = int(request.form['n_clusters'])
        session['n_clusters'] = n_clusters  # Store the number of clusters in the session

        # Reload the dataset from the session
        filepath = session.get('uploaded_file')
        if not filepath:
            return "No dataset found. Please upload a dataset first.", 400

        df, data = load_dataset(filepath, normalize=True)

        # Apply K-Means and generate the clustering visualization
        df_clustered, metrics_dict, kmeans_image = apply_kmeans_show_classes(df, data, n_clusters)

        # Return the partial template for K-Means results
        return render_template(
            'kmeans_partial.html',
            kmeans_image=kmeans_image,
            clustered_data=df_clustered.to_html(classes='table table-striped', index=False),
            metrics=metrics_dict,
            time=time.time
        )
    except Exception as e:
        return str(e), 500

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