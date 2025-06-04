#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CELL 1: Initialization, Model Loading, AND Comprehensive Unsupervised Clustering Analysis.
Includes K-Means, HDBSCAN, Elbow/Silhouette, and 2D visualization.
Displays image samples per cluster.

NOTE: This cell will run for a LONG TIME with k_range up to 50 and multiple algorithms.
"""

# --- Standard Library Imports ---
import os
import gc
import traceback
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

# Debug mode flag - set to False for production
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Optional matplotlib import
try:
    if DEBUG_MODE:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    else:
        MATPLOTLIB_AVAILABLE = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Debug logging
if DEBUG_MODE and not MATPLOTLIB_AVAILABLE:
    logger.warning("Debug mode is enabled but matplotlib is not available. Visualization will be disabled.")

# --- FastAPI Imports ---
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.http.models import SparseVector

# Initialize FastAPI app
app = FastAPI()

origins = [
    "https://image.ntopalli.com",
    "https://www.image.ntopalli.com",
    "http://localhost:3000"  # For local development
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Science & Machine Learning Imports ---
import numpy as np
import torch
from PIL import Image as PILImage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler # Good practice before PCA/Clustering
import pandas as pd # For easier handling of results

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️ HDBSCAN library not found. `pip install hdbscan`. HDBSCAN clustering will be skipped.")

# --- IPython & Notebook Imports ---
from IPython.display import Image as IPImage, HTML, clear_output, display

# --- Application Specific Imports ---
from transformers import AutoModel, AutoProcessor
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import FusionQuery, SparseVector as QdrantSparseVector

print("🚀 Initializing script... (This is CELL 1 content - see note above)")
print(f"🐍 Python version: {sys.version}")
# Version prints for key libraries
# ... (library version prints as before) ...

CACHE_DIR = Path("./models_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["FASTEMBED_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)
print(f"🛠️  Cache directory set to: {CACHE_DIR}")

# --- FastEmbed Availability Check ---
try:
    from fastembed import SparseTextEmbedding, TextEmbedding
    FASTEMBED_AVAILABLE = True
    print("✅ FastEmbed library specific imports successful.")
except ImportError as e_fe_import:
    FASTEMBED_AVAILABLE = False
    print(f"⚠️  FastEmbed specific imports FAILED: {e_fe_import}.")

# --- Configuration Constants ---
QDRANT_URL       = "https://eab9412a-9a65-4fcd-8e81-1bc153246739.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yQD0MKfTMJIlZjJEHpjBHxN-xeFkBNBKUq9LO5awyQg"
COLLECTION_NAME  = "hybrid_search_v13_siglip_e5_bge_m3"
QDRANT_TIMEOUT = 60

DENSE_IMAGE_VECTOR = "image_vector_siglip_so400m"
DENSE_TEXT_VECTOR  = "text_vector_e5_large"
SPARSE_TEXT_VECTOR = "text_vector_bm25"

IMAGE_PATH_COLUMN            = "matched_image_path"
TEXT_FOR_RANKING_PAYLOAD_KEY = "combined_metadata_text_ranked" # Used for text samples if needed
GEMINI_DESCRIPTION_KEY       = "gemini_image_description_de_200w"
ORIGINAL_UNIQUE_ID_COLUMN    = "unique_image_id"

# Search parameters (for the interactive search loop in Cell 2 - NOT changed here)
TOP_K_INITIAL_SEARCH  = 100
RRF_THRESHOLD_SEARCH  = 0.1

# --- Clustering parameters ---
VECTOR_NAME_FOR_CLUSTERING = DENSE_TEXT_VECTOR # Embeddings to use for clustering
MAX_POINTS_TO_CLUSTER = 1000 # START LOW (e.g. 500-1000). Increase cautiously! Max 50k for full run.
N_COMPONENTS_PCA = 50        # Target dimensions for PCA. If -1, PCA is skipped or var explained.
USE_PCA = True               # Whether to use PCA
K_RANGE_FOR_ELBOW_SILHOUETTE = range(2, 51) # Test k from 2 to 50 clusters

# HDBSCAN parameters
MIN_CLUSTER_SIZE_HDBSCAN = 10 # Min points in a cluster for HDBSCAN
MIN_SAMPLES_HDBSCAN = 5       # How conservative HDBSCAN is (None for default)

# Model names for embedding (Cell 1 init)
# Using CLIP instead of SigLIP to avoid sentencepiece dependency
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DENSE_TEXT_MODEL_NAME = "intfloat/multilingual-e5-large"
SPARSE_MODEL_NAME     = "Qdrant/bm25"

print("🛠️  [DEBUG] Constants defined (Clustering specific):")
print(f"    [DEBUG] VECTOR_NAME_FOR_CLUSTERING: {VECTOR_NAME_FOR_CLUSTERING}")
print(f"    [DEBUG] MAX_POINTS_TO_CLUSTER: {MAX_POINTS_TO_CLUSTER} << IMPORTANT FOR RUNTIME")
print(f"    [DEBUG] USE_PCA: {USE_PCA}, N_COMPONENTS_PCA: {N_COMPONENTS_PCA if USE_PCA else 'N/A'}")
print(f"    [DEBUG] K_RANGE_FOR_ELBOW_SILHOUETTE (KMeans): {K_RANGE_FOR_ELBOW_SILHOUETTE.start} to {K_RANGE_FOR_ELBOW_SILHOUETTE.stop -1}")
if HDBSCAN_AVAILABLE:
    print(f"    [DEBUG] HDBSCAN_MIN_CLUSTER_SIZE: {MIN_CLUSTER_SIZE_HDBSCAN}, MIN_SAMPLES: {MIN_SAMPLES_HDBSCAN}")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
device = "cpu"
print(f"🛠️  [DEBUG] Runtime environment: PyTorch device='{device}'.")
def _print_mem(tag=""): print(f"🧠 [DEBUG] Memory checkpoint: {tag}")

display(HTML(r"""<style> /* CSS as before */ </style>"""))
print("🎨 [DEBUG] Inline CSS for display applied.")

# --- Initialize Qdrant Client and Models (SigLIP, e5, BM25) ---
# ... (This part of the code remains IDENTICAL to your last working version for Cell 1 initialization)
print(f"🔌  [DEBUG] Connecting to Qdrant at {QDRANT_URL} with timeout {QDRANT_TIMEOUT}s...")
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    qdrant.get_collections()
    print("✅  [DEBUG] Successfully connected to Qdrant.")
except Exception as e:
    print(f"❌ FATAL: Could not connect to Qdrant: {e}"); traceback.print_exc(); raise SystemExit("Qdrant connection failed")

print(f"📦  [DEBUG] Loading CLIP model (HuggingFace Transformers): {CLIP_MODEL_NAME}...")
try:
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR).to(device).eval()
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, cache_dir=CACHE_DIR)
    print(f"✅  [DEBUG] CLIP model and processor loaded successfully. Device: {device}")
except Exception as e:
    print(f"❌ FATAL: Could not load CLIP model: {e}"); traceback.print_exc(); raise SystemExit("CLIP model loading failed")

sparse_model = None
if FASTEMBED_AVAILABLE:
    print(f"📦  [DEBUG] Loading BM25 sparse model (FastEmbed): “{SPARSE_MODEL_NAME}”...")
    try:
        sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, cache_dir=CACHE_DIR)
        _ = list(sparse_model.embed(["test bm25 query"]))
        print("✅  [DEBUG] BM25 sparse model loaded successfully via FastEmbed.")
    except Exception as e:
        print(f"❌ FATAL: Failed to load BM25 sparse model via FastEmbed: {e}"); traceback.print_exc(); raise SystemExit("BM25 FastEmbed model loading failed")
else:
    print(f"⚠️  [DEBUG] FastEmbed not available for BM25. Sparse search will be disabled if it relied on this.")

def _init_e5_embedder():
    _print_mem("[DEBUG] _init_e5_embedder called")
    if FASTEMBED_AVAILABLE:
        print(f"📦  [DEBUG] Initializing e5 dense text embedder (FastEmbed): “{DENSE_TEXT_MODEL_NAME}”...")
        try:
            fe_model = TextEmbedding(model_name=DENSE_TEXT_MODEL_NAME, cache_dir=CACHE_DIR)
            test_e5_emb = list(fe_model.embed(["hello e5 fastembed"]))
            dummy_emb_dim = test_e5_emb[0].shape[0] if test_e5_emb else "N/A"
            print(f"✅  [DEBUG] FastEmbed e5-large model “{DENSE_TEXT_MODEL_NAME}” loaded. Vector dim: {dummy_emb_dim}")
            return lambda texts: list(fe_model.embed(texts))
        except Exception as exc:
            print(f"❌ FATAL: FastEmbed e5 “{DENSE_TEXT_MODEL_NAME}” loading FAILED: {exc}"); traceback.print_exc(); raise SystemExit("FastEmbed e5 loading failed")
    else:
        print(f"📦  [DEBUG] FastEmbed not available for e5. Loading via ST: “{DENSE_TEXT_MODEL_NAME}”...")
        from sentence_transformers import SentenceTransformer # Import moved here
        try:
            st_model = SentenceTransformer(DENSE_TEXT_MODEL_NAME, cache_folder=CACHE_DIR, device=device)
            dummy_emb_st = st_model.encode(["hello e5 st"], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            dummy_emb_shape = dummy_emb_st.shape
            print(f"✅  [DEBUG] ST e5-large “{DENSE_TEXT_MODEL_NAME}” loaded. Dim: {dummy_emb_shape[-1] if len(dummy_emb_shape)>1 else 'N/A'}. Device: {device}")
            return lambda texts: st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"❌ FATAL: Could not load ST e5-large model: {e}"); traceback.print_exc(); raise SystemExit("ST e5 loading failed")
dense_embed = _init_e5_embedder()
_print_mem("[DEBUG] Core models for search initialized.")
# --- End of Standard Initialization ---


# --- Unsupervised Clustering Analysis Section ---
def perform_comprehensive_clustering_analysis():
    print("\n--- 🚀 Starting Comprehensive Unsupervised Clustering Analysis ---")
    _print_mem("[DEBUG] Clustering Analysis Start")

    print(f"[Clustering] Fetching up to {MAX_POINTS_TO_CLUSTER} points with '{VECTOR_NAME_FOR_CLUSTERING}' vectors from Qdrant...")
    all_points_for_clustering = [] # Stores Qdrant PointStruct objects
    try:
        scroll_result, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=MAX_POINTS_TO_CLUSTER,
            with_payload=True, with_vectors=[VECTOR_NAME_FOR_CLUSTERING] )
        all_points_for_clustering = scroll_result
        print(f"[Clustering] ✅ Fetched {len(all_points_for_clustering)} points from Qdrant.")
    except Exception as e_scroll:
        print(f"[Clustering] ❌ Error fetching data: {e_scroll}"); traceback.print_exc(); return

    if not all_points_for_clustering: print("[Clustering] No points fetched. Cannot cluster."); return

    point_ids = [p.id for p in all_points_for_clustering]
    # Extract vectors, ensuring they exist
    vectors_list = [p.vector[VECTOR_NAME_FOR_CLUSTERING] for p in all_points_for_clustering if p.vector and VECTOR_NAME_FOR_CLUSTERING in p.vector]
    if not vectors_list: print("[Clustering] ❌ No valid vectors for clustering in fetched points."); return
    
    vectors = np.array(vectors_list)
    if vectors.ndim == 1: vectors = vectors.reshape(-1, 1) if vectors.shape[0] == 1 and len(vectors_list[0]) == 1 else vectors.reshape(1, -1) # Adjust for single feature or single sample
    print(f"[Clustering] Vector array shape: {vectors.shape}")

    # --- Standardize data (good practice) ---
    print("[Clustering] Standardizing data...")
    scaler = StandardScaler()
    try:
        scaled_vectors = scaler.fit_transform(vectors)
    except ValueError as e_scale:
        print(f"[Clustering] ⚠️ StandardScaler failed: {e_scale}. Using original vectors. Might happen with very few samples/features.")
        scaled_vectors = vectors # Fallback to unscaled if scaler fails (e.g. single sample)

    data_to_cluster = scaled_vectors
    # --- PCA for dimensionality reduction and visualization ---
    # For visualization, we'll reduce to 2 components regardless of N_COMPONENTS_PCA for clustering
    pca_for_viz = None
    if data_to_cluster.shape[0] >= 2 and data_to_cluster.shape[1] >=2: # PCA needs at least 2 samples and 2 features
        print("[Clustering] Performing PCA for 2D visualization...")
        pca_for_viz = PCA(n_components=2, random_state=42)
        try:
            data_2d_viz = pca_for_viz.fit_transform(data_to_cluster) # Use scaled data for PCA
            print(f"[Clustering] ✅ PCA for visualization complete. Shape: {data_2d_viz.shape}")
        except ValueError as e_pca_viz:
            print(f"[Clustering] ⚠️ PCA for visualization failed: {e_pca_viz}. Visualization might not be possible.")
            data_2d_viz = None # Mark as None if PCA fails
            pca_for_viz = None
    else:
        print("[Clustering] PCA for visualization skipped (not enough samples/features).")
        data_2d_viz = None


    # Data for actual clustering can be different if N_COMPONENTS_PCA for clustering is different
    data_for_algo_clustering = data_to_cluster # Default to scaled vectors
    if USE_PCA and N_COMPONENTS_PCA > 0 and N_COMPONENTS_PCA < data_to_cluster.shape[1] and data_to_cluster.shape[0] > N_COMPONENTS_PCA:
        print(f"[Clustering] Performing PCA to {N_COMPONENTS_PCA} components for clustering algorithms...")
        pca_for_algo = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
        try:
            data_for_algo_clustering = pca_for_algo.fit_transform(data_to_cluster) # Use scaled data
            print(f"[Clustering] ✅ PCA for algorithms complete. Shape: {data_for_algo_clustering.shape}")
        except ValueError as e_pca_algo:
            print(f"[Clustering] ⚠️ PCA for algorithms failed: {e_pca_algo}. Using original scaled vectors for algorithms.")
            # data_for_algo_clustering remains scaled_vectors
    elif USE_PCA:
        print(f"[Clustering] PCA for algorithms skipped: Not enough samples/dimensions or N_COMPONENTS_PCA invalid.")


    _print_mem("[DEBUG] Data preprocessing for clustering complete")
    
    cluster_results: Dict[str, Dict[str, Any]] = {} # To store results of different algorithms

    # --- 1. K-Means with Elbow/Silhouette ---
    print("\n--- [Clustering Algorithm 1: K-Means] ---")
    print("[K-Means] Determining optimal k (Elbow/Silhouette)...")
    wcss_kmeans = []
    silhouette_kmeans_map = {}
    actual_k_range_kmeans = [k for k in K_RANGE_FOR_ELBOW_SILHOUETTE if k <= data_for_algo_clustering.shape[0]]

    if not actual_k_range_kmeans: print("[K-Means] ❌ Not enough samples for any k. Skipping K-Means k-determination.")
    else:
        for k_val in actual_k_range_kmeans:
            print(f"   [K-Means] Testing k={k_val}...")
            try:
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(data_for_algo_clustering)
                wcss_kmeans.append(kmeans.inertia_)
                if len(set(labels)) >= 2 and len(set(labels)) < len(labels):
                    score = silhouette_score(data_for_algo_clustering, labels); silhouette_kmeans_map[k_val] = score
                    print(f"      k={k_val}, Inertia: {kmeans.inertia_:.2f}, Silhouette: {score:.4f}")
                else: silhouette_kmeans_map[k_val] = -1; print(f"      k={k_val}, Inertia: {kmeans.inertia_:.2f}, Silhouette: N/A")
            except Exception as e: print(f"      Error k={k_val}: {e}"); wcss_kmeans.append(np.nan); silhouette_kmeans_map[k_val] = np.nan
        
        # Plot K-Means results
        if DEBUG_MODE and MATPLOTLIB_AVAILABLE:
            try:
                # Plot Elbow and Silhouette for KMeans
                plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(actual_k_range_kmeans, [w for w in wcss_kmeans if not np.isnan(w)], marker='o'); plt.title('K-Means: Elbow Method'); plt.xlabel('k'); plt.ylabel('WCSS'); plt.xticks(actual_k_range_kmeans);plt.grid(True)
                
                # Only plot silhouette if we have valid scores
                valid_s_k = [k for k, s in silhouette_kmeans_map.items() if s > -1 and not np.isnan(s)]
                valid_s_s = [s for k, s in silhouette_kmeans_map.items() if s > -1 and not np.isnan(s)]
                if valid_s_k: plt.subplot(1, 2, 2); plt.plot(valid_s_k, valid_s_s, marker='o'); plt.title('K-Means: Silhouette Scores'); plt.xlabel('k'); plt.ylabel('Avg Silhouette'); plt.xticks(actual_k_range_kmeans);plt.grid(True)
                
                plt.tight_layout(); plt.show(); display(plt.gcf()); plt.close()
            except Exception as e:
                logger.warning(f"Failed to generate K-means plots: {str(e)}")
        
        optimal_k_kmeans = -1
        if valid_s_k: optimal_k_kmeans = valid_s_k[np.argmax(valid_s_s)]; print(f"[K-Means] ✅ Optimal k by Silhouette: {optimal_k_kmeans} (Score: {max(valid_s_s):.4f})")
        else: optimal_k_kmeans = 3; print(f"[K-Means] ⚠️ No valid Silhouette. Defaulting k={optimal_k_kmeans}")

        if data_for_algo_clustering.shape[0] >= optimal_k_kmeans and optimal_k_kmeans > 0 :
            print(f"[K-Means] Running final K-Means with k={optimal_k_kmeans}...")
            final_kmeans = KMeans(n_clusters=optimal_k_kmeans, random_state=42, n_init='auto')
            kmeans_labels = final_kmeans.fit_predict(data_for_algo_clustering)
            cluster_results["KMeans"] = {"labels": kmeans_labels, "k": optimal_k_kmeans, "name": f"K-Means (k={optimal_k_kmeans})"}
            print(f"[K-Means] ✅ Final K-Means done. Found {len(set(kmeans_labels))} clusters.")
        else: print(f"[K-Means] ❌ Not enough samples for optimal_k={optimal_k_kmeans}. Skipping final K-Means.")


    # --- 2. HDBSCAN ---
    if HDBSCAN_AVAILABLE:
        print("\n--- [Clustering Algorithm 2: HDBSCAN] ---")
        try:
            print(f"[HDBSCAN] Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE_HDBSCAN}, min_samples={MIN_SAMPLES_HDBSCAN})...")
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE_HDBSCAN, min_samples=MIN_SAMPLES_HDBSCAN, metric='euclidean', allow_single_cluster=True) # allow_single_cluster might be needed for small datasets
            hdbscan_labels = hdbscan_clusterer.fit_predict(data_for_algo_clustering)
            n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0) # Exclude noise point label -1
            n_noise_hdbscan = np.sum(hdbscan_labels == -1)
            cluster_results["HDBSCAN"] = {"labels": hdbscan_labels, "k": n_clusters_hdbscan, "noise_points": n_noise_hdbscan, "name": f"HDBSCAN (found {n_clusters_hdbscan} clusters)"}
            print(f"[HDBSCAN] ✅ HDBSCAN complete. Found {n_clusters_hdbscan} clusters and {n_noise_hdbscan} noise points.")
        except Exception as e_hdb: print(f"[HDBSCAN] ❌ Error: {e_hdb}"); traceback.print_exc()
    else: print("\n--- [HDBSCAN Skipped (library not available)] ---")

    # --- 3. Agglomerative Clustering (Example, using optimal_k from KMeans) ---
    # print("\n--- [Clustering Algorithm 3: Agglomerative Clustering] ---")
    # if "KMeans" in cluster_results and cluster_results["KMeans"]["k"] > 0:
    #     k_for_agg = cluster_results["KMeans"]["k"]
    #     print(f"[Agglomerative] Running with n_clusters={k_for_agg} (from K-Means)...")
    #     try:
    #         agglomerative_model = AgglomerativeClustering(n_clusters=k_for_agg, linkage='ward')
    #         agglomerative_labels = agglomerative_model.fit_predict(data_for_algo_clustering)
    #         n_clusters_agg = len(set(agglomerative_labels))
    #         cluster_results["Agglomerative"] = {"labels": agglomerative_labels, "k": n_clusters_agg, "name": f"Agglomerative (k={n_clusters_agg})"}
    #         print(f"[Agglomerative] ✅ Complete. Found {n_clusters_agg} clusters.")
    #     except Exception as e_agg: print(f"[Agglomerative] ❌ Error: {e_agg}"); traceback.print_exc()
    # else: print("[Agglomerative] Skipped (KMeans optimal_k not available or invalid).")

    _print_mem("[DEBUG] All clustering algorithms run")

    # --- Display Image Samples for each algorithm's clusters ---
    print("\n--- [Clustering] Image Samples per Cluster ---")
    for algo_name, result_data in cluster_results.items():
        labels = result_data["labels"]
        num_algo_clusters = result_data["k"]
        print(f"\n  -- Algorithm: {result_data['name']} --")
        if num_algo_clusters == 0 and algo_name == "HDBSCAN": # Special case for HDBSCAN if it found no clusters
             print(f"     HDBSCAN found no actual clusters, only noise points ({result_data.get('noise_points',0)}). No samples to show.")
             continue
        if num_algo_clusters == 0 :
             print(f"     Algorithm found no clusters. No samples to show.")
             continue

        for i in range(num_algo_clusters): # Iterate up to num_algo_clusters
            cluster_label_to_check = i # For K-Means and Agglomerative, labels are 0 to k-1
            if algo_name == "HDBSCAN" and i == -1: continue # Skip noise points label for HDBSCAN display loop

            cluster_i_indices = [idx for idx, label in enumerate(labels) if label == cluster_label_to_check]
            if not cluster_i_indices:
                print(f"    Cluster {cluster_label_to_check}: Empty (this shouldn't happen if k is from set of labels)")
                continue
                
            print(f"    Cluster {cluster_label_to_check} (Size: {len(cluster_i_indices)}):")
            samples_to_show_imgs = min(2, len(cluster_i_indices)) # Show only 1-2 images per cluster to save output space
            
            html_img_display = []
            for sample_idx_loop in range(samples_to_show_imgs):
                original_point_index = cluster_i_indices[sample_idx_loop]
                if original_point_index < len(all_points_for_clustering):
                    point_obj = all_points_for_clustering[original_point_index]
                    payload = point_obj.payload or {}
                    img_path = payload.get(IMAGE_PATH_COLUMN)
                    if img_path and Path(img_path).is_file():
                        # Using IPImage for reliable display in Kaggle from any path Python can access.
                        # Limit width to reduce output cell size.
                        display(IPImage(filename=str(img_path), width=100, height=100)) # Small images
                    else:
                        print(f"      - Img for {point_obj.id}: Path missing or invalid ('{img_path}')")
                else: print(f"      - Error: Index out of bounds for sample in cluster {cluster_label_to_check}")
            if not samples_to_show_imgs : print("        No image samples to show for this cluster.")
            # A small pause after displaying images for a cluster
            time.sleep(0.1) 
            gc.collect()

    # --- 2D Visualization of Clusters (using PCA-reduced data to 2D if available) ---
    if data_2d_viz is not None and cluster_results:
        print("\n--- [Clustering] 2D Visualization of Clusters ---")
        
        # Choose which algorithm's labels to visualize (e.g., KMeans or HDBSCAN if it produced clusters)
        algo_to_visualize = "KMeans" if "KMeans" in cluster_results and cluster_results["KMeans"]["k"] > 0 else \
                            ("HDBSCAN" if "HDBSCAN" in cluster_results and cluster_results["HDBSCAN"]["k"] > 0 else None)

        if algo_to_visualize:
            labels_for_viz = cluster_results[algo_to_visualize]["labels"]
            k_for_viz = cluster_results[algo_to_visualize]["k"]
            
            if DEBUG_MODE and MATPLOTLIB_AVAILABLE:
                try:
                    # Plot clusters
                    plt.figure(figsize=(10, 8))
                    
                    # Get unique labels and sort them
                    unique_labels = sorted(np.unique(labels_for_viz))
                    
                    # Generate colors for each cluster
                    colors = plt.cm.get_cmap('viridis', len(unique_labels)) # Or 'tab20' for more colors
                    
                    # Plot each cluster
                    for i, label in enumerate(unique_labels):
                        if label == -1:
                            # Black used for noise.
                            points_in_cluster = data_2d_viz[labels_for_viz == label]
                            plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=10, color='grey', label='Noise', alpha=0.5)
                        else:
                            points_in_cluster = data_2d_viz[labels_for_viz == label]
                            plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], s=30, color=colors(i), label=f'Cluster {label}', alpha=0.7)
                    
                    plt.title(f'2D Visualization of Clusters ({cluster_results[algo_to_visualize]["name"]}) using PCA')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    if k_for_viz > 0 : plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.) # Only show legend if clusters exist
                    plt.grid(True)
                    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
                    plt.show()
                    display(plt.gcf())
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to generate cluster visualization: {str(e)}")
        else:
            print("[Clustering] No suitable clustering results to visualize in 2D.")
    else:
        print("[Clustering] 2D data for visualization not available (PCA to 2D might have failed or skipped).")

    # --- Qualitative Performance Comparison ---
    print("\n--- [Clustering] Qualitative Performance Summary ---")
    for algo_name, result_data in cluster_results.items():
        k = result_data.get("k", "N/A")
        name = result_data.get("name", algo_name)
        noise = result_data.get("noise_points", 0) if algo_name == "HDBSCAN" else 0
        
        silhouette_avg = "N/A"
        if algo_name == "KMeans" and k != "N/A" and k > 0 : # Calculate silhouette for the chosen k for KMeans
            if k in silhouette_kmeans_map and silhouette_kmeans_map[k] > -1:
                 silhouette_avg = f"{silhouette_kmeans_map[k]:.4f}"
        
        print(f"  Algorithm: {name}")
        print(f"    - Clusters Found: {k}")
        if algo_name == "HDBSCAN": print(f"    - Noise Points: {noise}")
        if silhouette_avg != "N/A": print(f"    - Avg Silhouette Score (for this k if KMeans): {silhouette_avg}")
        # Further qualitative notes could be added here based on visual inspection of samples / plots.
        # E.g., "Clusters appear well-separated" or "Many noise points with HDBSCAN"

    print("\n--- ավարտ Comprehensive Unsupervised Clustering Analysis ---")
    _print_mem("[DEBUG] Clustering Analysis End")
    # Cleanup large clustering variables
    del point_ids, vectors, scaled_vectors, data_for_algo_clustering, cluster_results, all_points_for_clustering
    if 'data_2d_viz' in locals() and data_2d_viz is not None: del data_2d_viz
    if 'pca_for_viz' in locals() and pca_for_viz is not None : del pca_for_viz
    if 'pca_for_algo' in locals() and 'pca_for_algo' in locals() and pca_for_algo is not None : del pca_for_algo
    if 'kmeans' in locals() and 'kmeans' in locals() and kmeans is not None : del kmeans # from k-determination loop
    if 'final_kmeans' in locals() and 'final_kmeans' in locals() and final_kmeans is not None : del final_kmeans
    if HDBSCAN_AVAILABLE and 'hdbscan_clusterer' in locals() and hdbscan_clusterer is not None : del hdbscan_clusterer
    gc.collect()

# --- Perform Clustering Analysis (Call the function) ---
if __name__ == '__main__' and '__file__' not in globals():
    RUN_CLUSTERING_ANALYSIS = True # Set to False to skip this long process
    if RUN_CLUSTERING_ANALYSIS:
        perform_comprehensive_clustering_analysis()
    else:
        print("ℹ️ Comprehensive Clustering analysis was SKIPPED based on RUN_CLUSTERING_ANALYSIS flag.")

print("✅✅✅ [DEBUG] INITIALIZATION AND CLUSTERING (IF ENABLED) COMPLETE. NEXT CELL FOR INTERACTIVE LOOP. ✅✅✅")
_print_mem("[DEBUG] End of Initialization & Clustering Cell")

# Cell 2 (Search Loop) would start here in a new Jupyter cell.
# The code for Cell 2 (the while True: loop for interactive search)
# should be your last stable version that worked for search WITHOUT clustering.

@app.on_event("startup")
async def startup_event():
    pass

@app.get("/search")
async def search_items(q: str = Query(None)):
    if not q:
        return []
    
    try:
        # Generate dense embedding using e5
        dense_vector = dense_embed([q])[0]
        
        # Generate sparse embedding using BM25
        sparse_vector = None
        if sparse_model:
            sparse_embed = list(sparse_model.embed([q]))[0]
            # Convert sparse embedding to the format expected by Qdrant
            if hasattr(sparse_embed, 'indices') and hasattr(sparse_embed, 'values'):
                sparse_vector = {
                    'indices': sparse_embed.indices.tolist() if hasattr(sparse_embed.indices, 'tolist') else list(sparse_embed.indices),
                    'values': sparse_embed.values.tolist() if hasattr(sparse_embed.values, 'tolist') else list(sparse_embed.values)
                }

        from collections import defaultdict
        
        # Function to perform search and normalize scores
        def perform_search(query_vector, vector_name, is_sparse=False):
            from qdrant_client import models
            
            if is_sparse:
                # For sparse vectors, use NamedSparseVector
                search_results = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=models.NamedSparseVector(
                        name=vector_name,
                        vector=models.SparseVector(
                            indices=query_vector['indices'],
                            values=query_vector['values']
                        )
                    ),
                    limit=TOP_K_INITIAL_SEARCH,
                    with_payload=True,
                    score_threshold=0.0
                )
            else:
                # For dense vectors, use regular named vector
                search_results = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=models.NamedVector(
                        name=vector_name,
                        vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
                    ),
                    limit=TOP_K_INITIAL_SEARCH,
                    with_payload=True,
                    score_threshold=0.0
                )
            
            # Normalize scores to 0-1 range
            if not search_results:
                return {}
                
            max_score = max(r.score for r in search_results) if search_results else 1.0
            normalized_results = {}
            for point in search_results:
                payload = point.payload or {}
                doc_id = payload.get(ORIGINAL_UNIQUE_ID_COLUMN)
                if not doc_id:
                    continue
                    
                normalized_score = point.score / max_score if max_score > 0 else 0
                normalized_results[doc_id] = {
                    "score": normalized_score,
                    "unique_image_id": doc_id,
                    "image_url": payload.get(IMAGE_PATH_COLUMN),
                    "description": payload.get(TEXT_FOR_RANKING_PAYLOAD_KEY),
                    "gemini_description": payload.get(GEMINI_DESCRIPTION_KEY)
                }
            return normalized_results
        
        # Perform both searches
        dense_results = perform_search(dense_vector, DENSE_TEXT_VECTOR, is_sparse=False)
        
        # Only perform sparse search if we have a sparse vector
        sparse_results = {}
        if sparse_vector:
            sparse_results = perform_search(sparse_vector, SPARSE_TEXT_VECTOR, is_sparse=True)
        
        # Combine results using RRF (Reciprocal Rank Fusion)
        combined_scores = defaultdict(float)
        all_doc_ids = set(dense_results.keys()) | set(sparse_results.keys())
        
        # Define weights for each search type
        weights = {
            'dense': 0.6,  # Higher weight for dense (semantic) search
            'sparse': 0.4   # Lower weight for sparse (keyword) search
        }
        
        # Calculate combined scores using RRF
        for doc_id in all_doc_ids:
            dense_rank = next((i+1 for i, (id_, _) in enumerate(
                sorted(dense_results.items(), key=lambda x: -x[1]['score'])) if id_ == doc_id), None)
            sparse_rank = next((i+1 for i, (id_, _) in enumerate(
                sorted(sparse_results.items(), key=lambda x: -x[1]['score'])) if id_ == doc_id), None)
            
            # Apply RRF formula: score = sum(1 / (k + rank))
            if dense_rank is not None:
                combined_scores[doc_id] += weights['dense'] * (1.0 / (60 + dense_rank))
            if sparse_rank is not None:
                combined_scores[doc_id] += weights['sparse'] * (1.0 / (60 + sparse_rank))
        
        # Get the final results
        all_ranked_results = []
        for doc_id, score in sorted(combined_scores.items(), key=lambda x: -x[1]):
            # Use the document from either search result (prefer dense as it has more context)
            doc = dense_results.get(doc_id) or sparse_results.get(doc_id)
            if doc:
                doc['score'] = score
                all_ranked_results.append(doc)
                
            if len(all_ranked_results) >= TOP_K_INITIAL_SEARCH: # Keep the original limit for total processed items
                break
        
        # Split into top 10 and others
        top_results_count = 10
        top_results = all_ranked_results[:top_results_count]
        other_results = all_ranked_results[top_results_count : top_results_count + 50]

        return {"top_results": top_results, "other_results": other_results}
    except Exception as e:
        print(f"Error performing search: {e}")
        return {"top_results": [], "other_results": []}

@app.get("/")
async def read_root():
    return {"Hello": "World"}
