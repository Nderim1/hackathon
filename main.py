from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer # For converting text query to vector
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import open_clip
from qdrant_client import QdrantClient, models
import time
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from urllib.parse import quote
import base64 # Für Base64-Encoding der Bilder
import io       # Für das Arbeiten mit Bytes im Speicher

load_dotenv() # Load environment variables from .env file


# Importiere scikit-learn für TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Importiere Sentence Transformers für Reranking
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("FEHLER: Die Bibliothek 'sentence-transformers' wurde nicht gefunden. Bitte installiere sie mit 'pip install -U sentence-transformers'.")
    exit()

# --- Konfiguration ---
QDRANT_URL = "https://eab9412a-9a65-4fcd-8e81-1bc153246739.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yQD0MKfTMJIlZjJEHpjBHxN-xeFkBNBKUq9LO5awyQg"
COLLECTION_NAME_TFIDF = "hybrid_image_search_kh_bm25_v1"

CSV_FILE_PATH_FOR_TFIDF_FIT = Path("/kaggle/working/enriched_metadata.csv")
IMAGE_PATH_COLUMN = "matched_image_path" # This should be the key in your Qdrant payload
ORIGINAL_UNIQUE_ID_COLUMN = "unique_image_id"
TEXT_FOR_SPARSE_COLUMN = "combined_metadata_text"
# a) Text für Reranking auf combined_metadata_text umstellen
TEXT_FOR_RERANKING_COLUMN = TEXT_FOR_SPARSE_COLUMN
LLM_DESCRIPTION_COLUMN = "gemini_image_description_de" # Bleibt für Anzeige erhalten

CLIP_MODEL_NAME = 'ViT-H-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s32b_b79k'

RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'

PREFETCH_LIMIT_DENSE = 75
PREFETCH_LIMIT_SPARSE_TFIDF = 100
FUSION_LIMIT_QDRANT = 60
PURE_SPARSE_TFIDF_LIMIT = 50
RERANK_TOP_N = 30
FINAL_TOP_RESULTS = 10
# b) Threshold etwas senken
RELEVANCE_THRESHOLD_SCALED = 1 # z.B. von 5 auf 1 gesenkt

DENSE_VECTOR_NAME = "image_vector_openclip"
SPARSE_VECTOR_NAME_TFIDF = "text_vector_tfidf_bm25"

PAYLOAD_FIELDS_TO_DISPLAY = {
    "Original ID": ORIGINAL_UNIQUE_ID_COLUMN, "Titel/Name": "name",
    "LLM Beschreibung": LLM_DESCRIPTION_COLUMN, # Wird immer noch angezeigt
    "Kombinierter Text (Reranking Basis)": TEXT_FOR_RERANKING_COLUMN, # Zeigt jetzt den Reranking Text
    "Bildnutzung": "Bildnutzung", "Original Dateiname": "file", "Copyright": "Copyright",
}
INTERNAL_PAYLOAD_NEEDS = [IMAGE_PATH_COLUMN, TEXT_FOR_RERANKING_COLUMN, ORIGINAL_UNIQUE_ID_COLUMN, TEXT_FOR_SPARSE_COLUMN, LLM_DESCRIPTION_COLUMN]

# --- Static File Configuration for Images ---
# IMPORTANT: Ensure this path matches where you download your images!
# Corrected based on `find_by_name` output
STATIC_IMAGE_DIR_PATH = "/Users/nde/projects/hackathon/kaggle/input/imaghes_khackin"
STATIC_IMAGE_URL_PREFIX = "/static_images_on_server" # URL path to access these images
# This is the prefix that appears to be incorrectly included in matched_image_path values from Qdrant
# Corrected based on server log output showing the paths from Qdrant use hyphen and double 'n'.
REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD = "/kaggle/input/imaghes-khackinn/"

tfidf_vectorizer_search = None
# ... (Rest der Initialisierungen bleibt gleich) ...
# --- Initialisierungen (gekürzt, da sie vorher funktionierten) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch verwendet: {device.upper()}")
# print("INFO: Initializing Qdrant client...") # Redundant, wird unten gemacht
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    # Ensure the correct collection name is used here for TF-IDF based collection
    qdrant_client.get_collection(collection_name=COLLECTION_NAME_TFIDF) 
    print(f"Erfolgreich mit Qdrant verbunden und Collection '{COLLECTION_NAME_TFIDF}' gefunden.")
except Exception as e: exit(f"FEHLER Qdrant: {e}")
clip_model, _, clip_preprocess = None,None,None
try:
    print(f"Lade OpenCLIP: {CLIP_MODEL_NAME}..."); clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED_DATASET, device=device); clip_model.eval(); print("OpenCLIP geladen.")
except Exception as e: exit(f"FEHLER OpenCLIP: {e}")
try:
    if CSV_FILE_PATH_FOR_TFIDF_FIT.exists():
        df_for_tfidf = pd.read_csv(CSV_FILE_PATH_FOR_TFIDF_FIT)
        if TEXT_FOR_SPARSE_COLUMN in df_for_tfidf.columns:
            corpus_texts_for_tfidf = df_for_tfidf[TEXT_FOR_SPARSE_COLUMN].astype(str).fillna('').tolist()
            if corpus_texts_for_tfidf:
                print(f"Trainiere TF-IDF auf {len(corpus_texts_for_tfidf)} Texten..."); tfidf_vectorizer_search = TfidfVectorizer(lowercase=True, stop_words=None, max_features=20000, sublinear_tf=True); tfidf_vectorizer_search.fit(corpus_texts_for_tfidf); print(f"TF-IDF trainiert. Vokabular: {len(tfidf_vectorizer_search.vocabulary_)}")
            else: print("WARN: Keine Texte für TF-IDF.")
        else: print(f"WARN: Spalte '{TEXT_FOR_SPARSE_COLUMN}' nicht in CSV für TF-IDF."); tfidf_vectorizer_search = None
    else: print(f"WARN: CSV für TF-IDF Training nicht gefunden."); tfidf_vectorizer_search = None
except Exception as e: print(f"FEHLER TF-IDF Init: {e}"); tfidf_vectorizer_search = None
reranker_model = None
try:
    print(f"Lade Reranker: {RERANKER_MODEL_NAME}..."); reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device, max_length=512); print("Reranker geladen.")
except Exception as e: print(f"FEHLER Reranker: {e}"); reranker_model = None

def get_clip_text_embedding(tq):
    if not clip_model: 
        print("DEBUG: get_clip_text_embedding returning None because clip_model is None")
        return None
    try:
        text_tokens = open_clip.tokenize([tq]).to(device)
        with torch.no_grad(): 
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"FEHLER in get_clip_text_embedding for query '{tq}': {e}")
        import traceback
        traceback.print_exc() 
        return None

def get_tfidf_sparse_vector_qdrant_format(tq):
    if not tfidf_vectorizer_search: return None
    try: m = tfidf_vectorizer_search.transform([tq]); return models.SparseVector(indices=m.indices.tolist(), values=m.data.tolist()) if m.nnz > 0 else None
    except Exception: return None
def image_to_base64(ip):
    try: img = Image.open(ip); fmt = img.format or 'JPEG'; mt = f"image/{fmt.lower().replace('jpeg','jpg')}"; mt = "image/jpeg" if mt=="image/jpg" else mt; buf = io.BytesIO(); img.save(buf, format=fmt); return f"data:{mt};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except: return None
def display_results_html(df_r):
    if df_r.empty: print("Keine Ergebnisse."); return
    h = "<div style='font-family:sans-serif;'>"
    for _, r in df_r.iterrows():
        h += "<div style='border:1px solid #ddd;padding:15px;margin-bottom:20px;display:flex;align-items:flex-start;box-shadow:2px 2px 5px #eee;'>"
        ips = r.get(IMAGE_PATH_COLUMN)
        if pd.notna(ips) and Path(ips).exists(): b64 = image_to_base64(ips); h += f"<div style='margin-right:20px;flex-shrink:0;'><img src='{b64 or ''}' alt='{Path(ips).name}' style='max-width:220px;max-height:220px;object-fit:contain;'/><p style='font-size:0.8em;text-align:center;'>{Path(ips).name}</p></div>" if b64 else "<div style='margin-right:20px;width:220px;height:220px;background-color:#f0f0f0;display:flex;align-items:center;justify-content:center;'>Bild nicht ladbar</div>"
        else: h += "<div style='margin-right:20px;width:220px;height:220px;background-color:#f0f0f0;display:flex;align-items:center;justify-content:center;'>Pfad fehlt</div>"
        mh = f"<div style='flex-grow:1;'><h4 style='margin-top:0;'>ID: {r.get(ORIGINAL_UNIQUE_ID_COLUMN,'N/A')} (Score: {r.get('relevance_score_scaled',0.0):.1f})</h4>";
        for lbl, col in PAYLOAD_FIELDS_TO_DISPLAY.items():
            v = r.get(col)
            if pd.notna(v) and str(v).strip(): txt = str(v); mh += f"<p style='font-size:0.9em;margin:4px 0;'><strong>{lbl}:</strong> {txt[:150]+'...' if len(txt)>150 else txt}</p>"
        h += mh + "</div></div>"
    return h + "</div>"

# --- Suchfunktion für REINE SPARSE (TF-IDF) SUCHE ---
def perform_pure_tfidf_search(query_text):
    print(f"Reine Sparse-Suche (TF-IDF/BM25-ähnlich) nach: '{query_text}'...")
    if not query_text.strip(): print("Bitte Suchbegriff eingeben."); return pd.DataFrame()
    if not tfidf_vectorizer_search: print("TF-IDF Vektorisierer nicht initialisiert."); return pd.DataFrame()
    query_sparse_tfidf_vector = get_tfidf_sparse_vector_qdrant_format(query_text)
    if not query_sparse_tfidf_vector: print("Konnte keinen TF-IDF Sparse-Vektor für Query generieren."); return pd.DataFrame()
    
    retrieved_hits = []
    try:
        print(f"Führe reine TF-IDF Suche in Qdrant (Col: {COLLECTION_NAME_TFIDF}, Limit: {PURE_SPARSE_TFIDF_LIMIT})...");
        qr = qdrant_client.query_points(COLLECTION_NAME_TFIDF, query=query_sparse_tfidf_vector, using=SPARSE_VECTOR_NAME_TFIDF, limit=PURE_SPARSE_TFIDF_LIMIT, with_payload=True, with_vectors=False)
        retrieved_hits = qr.points; print(f"{len(retrieved_hits)} Qdrant TF-IDF Treffer.")
    except Exception as e: print(f"FEHLER reine TF-IDF Suche: {e}"); return pd.DataFrame()
    if not retrieved_hits: print("Keine Qdrant TF-IDF Treffer."); return pd.DataFrame()
    
    valid_candidates = []
    for hit in retrieved_hits[:RERANK_TOP_N]:
        p=hit.payload; tr=p.get(TEXT_FOR_RERANKING_COLUMN) # TEXT_FOR_RERANKING_COLUMN jetzt combined_metadata_text
        if p and pd.notna(tr) and str(tr).strip(): valid_candidates.append(((hit.id,p.get(ORIGINAL_UNIQUE_ID_COLUMN,hit.id)),hit.score,p,str(tr)))
    
    rerank_pairs = [(query_text,c[3]) for c in valid_candidates]
    processed_results = []
    if rerank_pairs and reranker_model:
        print(f"Reranke {len(rerank_pairs)} Kandidaten...")
        MAX_RERANKER_TEXT_LENGTH = 1000 # Characters, as a proxy for tokens. Adjust as needed.
        truncated_rerank_pairs = []
        for q, doc_text in rerank_pairs:
            truncated_rerank_pairs.append((q, doc_text[:MAX_RERANKER_TEXT_LENGTH]))
        
        print(f"DEBUG: Rerank pairs (first 3, doc truncated to {MAX_RERANKER_TEXT_LENGTH} chars): {truncated_rerank_pairs[:3]}")
        try:
            raw_scores = reranker_model.predict(truncated_rerank_pairs,show_progress_bar=False,convert_to_tensor=True)
            print(f"DEBUG: Reranked raw scores (before sigmoid, first 10): {raw_scores[:10]}")
            scores_tensor = torch.sigmoid(raw_scores)
            
            # Handle potential NaNs from sigmoid or model directly
            scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0) # Replace NaN with 0.0
            
            scores = scores_tensor.cpu().numpy()
            print(f"DEBUG: Reranked scores (sigmoid applied, NaN replaced, first 10): {scores[:10]}")
            
            for i,((qid,oid),_,p,_) in enumerate(valid_candidates):
                image_filename_from_qdrant = p.get(IMAGE_PATH_COLUMN)
                actual_image_path_relative_to_static_dir = image_filename_from_qdrant
                if image_filename_from_qdrant and image_filename_from_qdrant.startswith(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):
                    actual_image_path_relative_to_static_dir = image_filename_from_qdrant[len(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):]
                
                image_url = f"{STATIC_IMAGE_URL_PREFIX}/{actual_image_path_relative_to_static_dir}" if actual_image_path_relative_to_static_dir else None
                processed_results.append({
                    "qdrant_id_uuid":qid,
                    **p,
                    "relevance_score_raw":float(scores[i]),
                    "relevance_score_scaled":float(scores[i]*100),
                    "image_url": image_url
                })
            processed_results.sort(key=lambda x: x["relevance_score_raw"],reverse=True); print("Reranking fertig.")
        except Exception as e: 
            print(f"Fehler Reranking: {e}. Nutze Qdrant Scores.")
            import traceback
            traceback.print_exc()
            processed_results=[{"qdrant_id_uuid":c[0][0],**c[2],"relevance_score_raw":c[1],"relevance_score_scaled":c[1]*100} for c in valid_candidates]
    else:
        print("Kein Reranking.")
        for h in retrieved_hits:
            payload = h.payload or {}
            image_filename_from_qdrant = payload.get(IMAGE_PATH_COLUMN)
            actual_image_path_relative_to_static_dir = image_filename_from_qdrant
            if image_filename_from_qdrant and image_filename_from_qdrant.startswith(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):
                actual_image_path_relative_to_static_dir = image_filename_from_qdrant[len(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):]
                
            image_url = f"{STATIC_IMAGE_URL_PREFIX}/{actual_image_path_relative_to_static_dir}" if actual_image_path_relative_to_static_dir else None
            processed_results.append({
                "qdrant_id_uuid": h.id,
                **payload,
                "relevance_score_raw": h.score,
                "relevance_score_scaled": h.score * 100,
                "image_url": image_url
            })
 
    if not processed_results: print("Keine Ergebnisse nach Verarbeitung."); return pd.DataFrame()
    df_res = pd.DataFrame(processed_results)
    
    # c) Erweiterter Fallback
    results_df_final = pd.DataFrame([])
    if not df_res.empty:
        results_df_filtered_by_threshold = df_res[df_res["relevance_score_scaled"] >= RELEVANCE_THRESHOLD_SCALED]
        if not results_df_filtered_by_threshold.empty:
            results_df_final = results_df_filtered_by_threshold.head(FINAL_TOP_RESULTS)
            print(f"{len(results_df_final)} finale Ergebnisse nach Threshold ({RELEVANCE_THRESHOLD_SCALED}/100).")
        else: # Threshold nicht erreicht
            print(f"Keine Ergebnisse über Threshold ({RELEVANCE_THRESHOLD_SCALED}/100). Zeige Top {FINAL_TOP_RESULTS} (falls vorhanden) der rerankten/Qdrant-Liste.")
            results_df_final = df_res.head(FINAL_TOP_RESULTS)
    
    if results_df_final.empty and retrieved_hits and df_res.empty: # Fallback, falls processed_results leer war, aber Qdrant Treffer hatte
        print(f"WARNUNG: processed_results war leer. Zeige Top {FINAL_TOP_RESULTS} der rohen Qdrant-Ergebnisse.")
        temp_fallback = [{"qdrant_id_uuid": h.id, **(h.payload or {}), "relevance_score_raw": h.score, "relevance_score_scaled": h.score*100} for h in retrieved_hits]
        results_df_final = pd.DataFrame(temp_fallback).head(FINAL_TOP_RESULTS)
            
    return results_df_final


# --- Suchfunktion für HYBRIDE SUCHE (Dense + BM25 TF-IDF) ---
def perform_hybrid_bm25_search(query_text):
    start_time_total = time.time()
    print(f"Hybride Suche (Dense + BM25 TF-IDF) nach: '{query_text}'...")

    if not query_text.strip(): print("Bitte Suchbegriff eingeben."); return pd.DataFrame()

    query_dense_vector = get_clip_text_embedding(query_text)
    query_sparse_tfidf_vector = get_tfidf_sparse_vector_qdrant_format(query_text)

    if not query_dense_vector and not tfidf_vectorizer_search: print("Kein CLIP & kein TF-IDF."); return pd.DataFrame()
    if not query_dense_vector and not query_sparse_tfidf_vector: print("Keine Query-Vektoren."); return pd.DataFrame()
    
    prefetches = []
    if query_dense_vector: prefetches.append(models.Prefetch(query=query_dense_vector, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT_DENSE))
    if query_sparse_tfidf_vector and tfidf_vectorizer_search: prefetches.append(models.Prefetch(query=query_sparse_tfidf_vector, using=SPARSE_VECTOR_NAME_TFIDF, limit=PREFETCH_LIMIT_SPARSE_TFIDF))

    retrieved_hits = []
    if prefetches:
        try:
            print(f"Führe hybride Qdrant-Suche (Col: {COLLECTION_NAME_TFIDF})...");
            qr = qdrant_client.query_points(COLLECTION_NAME_TFIDF, prefetch=prefetches, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=FUSION_LIMIT_QDRANT, with_payload=True, with_vectors=False)
            retrieved_hits = qr.points; print(f"{len(retrieved_hits)} Qdrant RRF Treffer.")
        except Exception as e: print(f"FEHLER Qdrant Hybrid: {e}"); return pd.DataFrame()
    else: print("Keine Prefetches für Query."); return pd.DataFrame()
    if not retrieved_hits: print("Keine Qdrant RRF Treffer."); return pd.DataFrame()
    
    valid_candidates = []
    # Using TEXT_FOR_RERANKING_COLUMN which is defined globally, e.g., 'combined_metadata_text'
    for hit in retrieved_hits[:RERANK_TOP_N]: # RERANK_TOP_N should limit candidates before reranking
        p=hit.payload
        text_for_rerank = p.get(TEXT_FOR_RERANKING_COLUMN)
        if p and pd.notna(text_for_rerank) and str(text_for_rerank).strip():
            valid_candidates.append(((hit.id,p.get(ORIGINAL_UNIQUE_ID_COLUMN,hit.id)),hit.score,p,str(text_for_rerank)))
        else:
            print(f"DEBUG: Skipping candidate for reranking due to missing/empty '{TEXT_FOR_RERANKING_COLUMN}': ID {hit.id}")

    processed_results = []
    if valid_candidates and reranker_model:
        rerank_pairs = [(query_text, candidate_data[3]) for candidate_data in valid_candidates]
        print(f"DEBUG: Reranking {len(rerank_pairs)} valid candidates...")
        
        MAX_RERANKER_TEXT_LENGTH = 512 # Truncate document text for reranker model if too long
        truncated_rerank_pairs = []
        for q, doc_text in rerank_pairs:
            truncated_doc_text = doc_text[:MAX_RERANKER_TEXT_LENGTH]
            if len(doc_text) > MAX_RERANKER_TEXT_LENGTH:
                print(f"DEBUG: Truncated doc text from {len(doc_text)} to {len(truncated_doc_text)} chars for reranker.")
            truncated_rerank_pairs.append((q, truncated_doc_text))
        
        print(f"DEBUG: Sample rerank_pairs (query, doc text truncated to {MAX_RERANKER_TEXT_LENGTH} chars, first 3): {truncated_rerank_pairs[:3]}")
        
        try:
            # Get raw scores from the model (these are logits, not probabilities yet)
            model_raw_scores_tensor = reranker_model.predict(truncated_rerank_pairs,show_progress_bar=False,convert_to_tensor=True)
            print(f"DEBUG: Raw scores from reranker model (logits, first 10): {model_raw_scores_tensor.tolist()[:10]}")
            
            # Apply sigmoid to convert logits to probabilities (0 to 1)
            probabilities_tensor = torch.sigmoid(model_raw_scores_tensor)
            print(f"DEBUG: Probabilities after sigmoid (first 10): {probabilities_tensor.tolist()[:10]}")
            
            # Handle potential NaNs that might arise from model or sigmoid (though less likely with sigmoid on reals)
            # nan_to_num will convert NaN to 0.0, +inf to large finite, -inf to small finite.
            probabilities_tensor_no_nan = torch.nan_to_num(probabilities_tensor, nan=0.0)
            print(f"DEBUG: Probabilities after sigmoid & nan_to_num(0.0) (first 10): {probabilities_tensor_no_nan.tolist()[:10]}")
            
            final_scores_for_items = probabilities_tensor_no_nan.cpu().numpy()
            
            for i, ((qdrant_id, original_id), qdrant_score, payload_dict, doc_text_used) in enumerate(valid_candidates):
                image_filename_from_qdrant = payload_dict.get(IMAGE_PATH_COLUMN)
                actual_image_path_relative_to_static_dir = image_filename_from_qdrant
                if image_filename_from_qdrant and image_filename_from_qdrant.startswith(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):
                    actual_image_path_relative_to_static_dir = image_filename_from_qdrant[len(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):]
                
                image_url = f"{STATIC_IMAGE_URL_PREFIX}/{actual_image_path_relative_to_static_dir}" if actual_image_path_relative_to_static_dir else None
                
                # 'relevance_score_raw' is the probability (0-1) after sigmoid and NaN handling
                # 'relevance_score_scaled' is this probability * 100
                current_score_raw = float(final_scores_for_items[i])
                current_score_scaled = current_score_raw * 100

                processed_results.append({
                    "qdrant_id_uuid": qdrant_id,
                    **payload_dict,
                    "relevance_score_raw": current_score_raw,
                    "relevance_score_scaled": current_score_scaled,
                    "image_url": image_url
                })
            processed_results.sort(key=lambda x: x["relevance_score_raw"], reverse=True)
            print("DEBUG: Reranking processing finished.")

        except Exception as e: 
            print(f"ERROR: Exception during reranking: {e}. Falling back to Qdrant scores.")
            import traceback
            traceback.print_exc()
            # Fallback to Qdrant scores if reranking fails
            processed_results = [] # Clear any partial results
            for ((qdrant_id, original_id), qdrant_score, payload_dict, doc_text_used) in valid_candidates:
                image_filename_from_qdrant = payload_dict.get(IMAGE_PATH_COLUMN)
                actual_image_path_relative_to_static_dir = image_filename_from_qdrant
                if image_filename_from_qdrant and image_filename_from_qdrant.startswith(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):
                    actual_image_path_relative_to_static_dir = image_filename_from_qdrant[len(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):]
                image_url = f"{STATIC_IMAGE_URL_PREFIX}/{actual_image_path_relative_to_static_dir}" if actual_image_path_relative_to_static_dir else None
                processed_results.append({
                    "qdrant_id_uuid": qdrant_id, 
                    **payload_dict, 
                    "relevance_score_raw": qdrant_score, # Use Qdrant score as raw
                    "relevance_score_scaled": qdrant_score * 100, # Scale Qdrant score
                    "image_url": image_url
                })
    else:
        print("DEBUG: No reranking performed (no valid candidates or no model). Using Qdrant scores.")
        for h in retrieved_hits: # This loop processes all retrieved_hits if no reranking
            payload = h.payload or {}
            image_filename_from_qdrant = payload.get(IMAGE_PATH_COLUMN)
            actual_image_path_relative_to_static_dir = image_filename_from_qdrant
            if image_filename_from_qdrant and image_filename_from_qdrant.startswith(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):
                actual_image_path_relative_to_static_dir = image_filename_from_qdrant[len(REDUNDANT_PATH_PREFIX_IN_QDRANT_PAYLOAD):]
            image_url = f"{STATIC_IMAGE_URL_PREFIX}/{actual_image_path_relative_to_static_dir}" if actual_image_path_relative_to_static_dir else None
            processed_results.append({
                "qdrant_id_uuid": h.id,
                **payload,
                "relevance_score_raw": h.score, # Use Qdrant score as raw
                "relevance_score_scaled": h.score * 100, # Scale Qdrant score
                "image_url": image_url
            })
 
    if not processed_results: print("Keine Ergebnisse nach Verarbeitung."); return pd.DataFrame()
    df_res = pd.DataFrame(processed_results)

    results_df_final = pd.DataFrame([])
    if not df_res.empty:
        results_df_filtered_by_threshold = df_res[df_res["relevance_score_scaled"] >= RELEVANCE_THRESHOLD_SCALED]
        if not results_df_filtered_by_threshold.empty:
            results_df_final = results_df_filtered_by_threshold.head(FINAL_TOP_RESULTS)
            print(f"{len(results_df_final)} finale Ergebnisse nach Threshold ({RELEVANCE_THRESHOLD_SCALED}/100).")
        else:
            print(f"Keine Ergebnisse über Threshold ({RELEVANCE_THRESHOLD_SCALED}/100). Zeige Top {FINAL_TOP_RESULTS} (falls vorhanden) der rerankten/Qdrant-Liste.")
            results_df_final = df_res.head(FINAL_TOP_RESULTS)
    
    if results_df_final.empty and not df_res.empty:
        print(f"Da auch die Top-Ergebnisse (vor Threshold) leer sind oder nicht existieren, zeige Top {FINAL_TOP_RESULTS} der Qdrant Ergebnisse (falls vorhanden).")
        # Fallback to df_res if results_df_final is empty but df_res is not (this case is already covered by the line above)
        # The following if block seems redundant with the logic above it.
        # if df_res.empty and retrieved_hits:
        #     print("WARNUNG: processed_results war leer, obwohl Qdrant Treffer hatte. Zeige rohe Qdrant-Treffer.")
        #     temp_fallback = [{"qdrant_id_uuid": h.id, **(h.payload or {}), "relevance_score_raw": h.score, "relevance_score_scaled": h.score*100} for h in retrieved_hits]
        #     results_df_final = pd.DataFrame(temp_fallback).head(FINAL_TOP_RESULTS)
            
    print("DEBUG: Final results_df_final head before returning from perform_hybrid_bm25_search:")
    # Ensure columns exist before trying to print, to prevent KeyError if df is empty or columns missing
    cols_to_print = ['qdrant_id_uuid', 'relevance_score_raw', 'relevance_score_scaled', 'image_url']
    existing_cols_to_print = [col for col in cols_to_print if col in results_df_final.columns]
    if not results_df_final.empty and existing_cols_to_print:
        print(results_df_final[existing_cols_to_print].head().to_string())
    elif results_df_final.empty:
        print("DEBUG: results_df_final is empty before returning.")
    else:
        print("DEBUG: results_df_final is not empty but lacks some expected columns for printing.")

    return results_df_final

app = FastAPI()

# CORS Middleware (Beispiel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Erlaube alle Origins (für Entwicklung)
    allow_credentials=True,
    allow_methods=["*"], # Erlaube alle Methoden
    allow_headers=["*"], # Erlaube alle Header
)

# Mount static directory for images
# Ensure the directory STATIC_IMAGE_DIR_PATH exists and contains your images.
# The images will be accessible at http://localhost:8000/STATIC_IMAGE_URL_PREFIX/your_image_filename.jpg
try:
    Path(STATIC_IMAGE_DIR_PATH).mkdir(parents=True, exist_ok=True) # Ensure directory exists
    app.mount(STATIC_IMAGE_URL_PREFIX, StaticFiles(directory=STATIC_IMAGE_DIR_PATH), name="static_images_on_server")
    print(f"Serving static images from '{STATIC_IMAGE_DIR_PATH}' at '{STATIC_IMAGE_URL_PREFIX}'")
except Exception as e:
    print(f"FEHLER beim Mounten des Static Image Verzeichnisses: {e}")

@app.on_event("startup")
async def startup_event():
    pass

@app.get("/search")
async def search_items(q: str = Query(None)):
    # Updated to call the new hybrid search function
    results_df = perform_hybrid_bm25_search(q)
    if results_df is None or results_df.empty:
        return []
    # Convert DataFrame to list of dicts for JSON response
    return results_df.to_dict(orient='records')

@app.get("/")
async def read_root():
    return {"Hello": "World"}
