# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script contains helper functions for calculating initial embeddings
# using local and global heuristics.


import torch
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
import faiss
from typing import List, Optional,Tuple
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from typing import Optional, Tuple, Dict, Set

def calculate_global_embedding(
    query_token_str: str,
    full_token_embeds_cache: dict,
    faiss_index: faiss.Index,
    faiss_id_to_old_vocab_id: Dict[int, int],
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    k: int,
    temperature: float,
    data_type: torch.dtype,
    device: str = "cpu",
    valid_faiss_indices_set: Optional[Set[int]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embeddings using the Global heuristic with optimized performance.

    Args:
        query_token_str: String representation of the new token (decoded).
        full_token_embeds_cache: Cache mapping token strings to external embeddings.
        faiss_index: Pre-built FAISS index of old vocabulary embeddings (e.g., IndexFlatIP).
        faiss_id_to_old_vocab_id: Mapping from FAISS index IDs to original vocab IDs.
        valid_faiss_indices_set: Optional set of valid FAISS IDs; derived if None.
        original_input_embeddings: Input embedding matrix of the original model.
        original_output_embeddings: Output embedding matrix if untied; None otherwise.
        k: Number of nearest neighbors to retrieve.
        temperature: Softmax temperature for weighting similarities.
        data_type: Torch data type for calculations (e.g., torch.float32).
        device: Device for tensor operations (e.g., 'cuda', 'cpu').
    
    Returns:
        Tuple of (input_embedding, output_embedding), both on CPU or None if failed.
    """
    # Derive valid FAISS indices if not provided
    if valid_faiss_indices_set is None:
        valid_faiss_indices_set = set(faiss_id_to_old_vocab_id.keys())

    # Check cache and return early if token is missing
    query_embedding_list = full_token_embeds_cache.get(query_token_str)
    if query_embedding_list is None:
        return None, None

    try:
        # Prepare and normalize query embedding
        query_embedding = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Perform FAISS search for nearest neighbors
        distances, indices = faiss_index.search(query_embedding, k)
        neighbor_faiss_indices = indices[0]
        neighbor_scores = distances[0]

        # Filter valid neighbors using precomputed mapping
        valid_old_vocab_ids = []
        valid_scores = []
        for i, faiss_idx in enumerate(neighbor_faiss_indices):
            if faiss_idx != -1 and faiss_idx in valid_faiss_indices_set:
                old_vocab_id = faiss_id_to_old_vocab_id[faiss_idx]
                valid_old_vocab_ids.append(old_vocab_id)
                valid_scores.append(neighbor_scores[i])

        if not valid_old_vocab_ids:
            return None, None

        # Compute weights; handle single-neighbor edge case
        scores_tensor = torch.tensor(valid_scores, device=device, dtype=data_type)
        weights = torch.ones(1, device=device, dtype=data_type) if scores_tensor.numel() == 1 else \
                 F.softmax(scores_tensor / temperature, dim=0)
        weights = weights.unsqueeze(1)

        # Calculate embeddings directly on target device
        valid_ids_tensor = torch.tensor(valid_old_vocab_ids, device=device, dtype=torch.long)
        neighbor_input_embeds = original_input_embeddings.index_select(0, valid_ids_tensor).to(dtype=data_type)
        global_embedding_input = (weights * neighbor_input_embeds).sum(dim=0)

        global_embedding_output = None
        if original_output_embeddings is not None:
            neighbor_output_embeds = original_output_embeddings.index_select(0, valid_ids_tensor).to(dtype=data_type)
            global_embedding_output = (weights * neighbor_output_embeds).sum(dim=0)

        # Move results to CPU
        return global_embedding_input.cpu(), global_embedding_output.cpu() if global_embedding_output is not None else None

    except Exception as e:
        print(f"Error calculating global embedding for token '{query_token_str}': {e}")
        return None, None
    

def precompute_subtoken_embeddings_array(
    old_tokenizer: AutoTokenizer,
    subtoken_embeds_cache: Dict[str, List[float]],
    embedding_dim: int,
    max_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputes a NumPy array of embeddings indexed by old token IDs.
    
    Args:
        old_tokenizer: Tokenizer for the old vocabulary.
        subtoken_embeds_cache: Cache mapping subtoken strings to embeddings.
        embedding_dim: Dimension of the embeddings.
        max_id: Maximum ID to consider (e.g., embedding matrix size).
        
    Returns:
        Tuple of (embeddings_array, valid_mask) where embeddings_array is float32 and valid_mask is bool.
    """
    embeddings_array = np.zeros((max_id, embedding_dim), dtype=np.float32)
    valid_mask = np.zeros(max_id, dtype=bool)
    
    vocab = old_tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        if 0 <= token_id < max_id:
            embedding = subtoken_embeds_cache.get(token_str)
            if embedding is not None:
                embeddings_array[token_id] = embedding
                valid_mask[token_id] = True
    
    print(f"Precomputed embeddings for {valid_mask.sum()} out of {max_id} IDs.")
    return embeddings_array, valid_mask

def calculate_local_embedding_optimized_array(
    token_str: str,
    new_token_id: int,
    new_tokenizer: AutoTokenizer,
    old_tokenizer: AutoTokenizer,
    full_token_embeds_cache: Dict[str, List[float]],
    subtoken_embeddings_array: np.ndarray,
    subtoken_valid_mask: np.ndarray,
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    temperature: float,
    data_type: torch.dtype,
    device: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embeddings using the Local Subword Composition heuristic with array-based optimizations.
    
    Args:
        token_str: Unique token string from the new vocabulary.
        new_token_id: ID of the token in the new vocabulary (for decoding if needed).
        new_tokenizer: Tokenizer for the new vocabulary.
        old_tokenizer: Tokenizer for the old vocabulary.
        full_token_embeds_cache: Cache of token strings to external embeddings.
        subtoken_embeddings_array: Precomputed array of embeddings indexed by token ID.
        subtoken_valid_mask: Boolean mask indicating valid token IDs in the array.
        original_input_embeddings: Input embedding matrix of the original model.
        original_output_embeddings: Output embedding matrix if untied; None otherwise.
        temperature: Softmax temperature for weighting similarities.
        data_type: Torch data type for calculations (e.g., torch.float32).
        device: Device for tensor operations (e.g., 'cuda', 'cpu').
        
    Returns:
        Tuple of (input_embedding, output_embedding) on CPU, or (None, None) if failed.
    """
    try:
        # Decode token and fetch its embedding
        full_token_decoded = token_str or new_tokenizer.decode([new_token_id], clean_up_tokenization_spaces=False)
        full_embed_list = full_token_embeds_cache.get(full_token_decoded)
        if not full_embed_list:
            return None, None
        
        full_embed_ext = torch.tensor(full_embed_list, dtype=data_type, device=device)
        
        # Tokenize into subtokens with old tokenizer
        old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
        if not old_ids:
            return None, None
        
        # Vectorized filtering of valid subtokens using precomputed mask
        old_ids_array = np.array(old_ids, dtype=np.int64)
        mask = (old_ids_array >= 0) & (old_ids_array < subtoken_embeddings_array.shape[0]) & subtoken_valid_mask[old_ids_array]
        valid_old_ids = old_ids_array[mask]
        
        if not valid_old_ids.size:
            return None, None
        
        # Direct array lookup instead of dictionary access
        valid_subtoken_embeds = subtoken_embeddings_array[valid_old_ids]
        valid_subtoken_strs = [old_tokenizer.decode([int(oid)], clean_up_tokenization_spaces=False) for oid in valid_old_ids]
        
        # Compute weights: cosine similarity and length normalization
        sub_embeds_ext = torch.tensor(valid_subtoken_embeds, dtype=data_type, device=device)
        similarities = F.cosine_similarity(full_embed_ext.unsqueeze(0), sub_embeds_ext, dim=1)
        weights_sim = F.softmax(similarities / temperature, dim=0)
        
        len_full = max(len(full_token_decoded), 1)  # Avoid division by zero
        len_norm_values = np.array([len(s) / len_full for s in valid_subtoken_strs], dtype=np.float32)
        len_norm = torch.from_numpy(len_norm_values).to(device=device, dtype=data_type)
        
        combined_weights = (weights_sim + len_norm) / 2.0
        final_weights = F.softmax(combined_weights / (temperature + 1e-9), dim=0).unsqueeze(1)
        
        # Calculate weighted embeddings
        valid_ids_tensor = torch.from_numpy(valid_old_ids).to(device=device, dtype=torch.long)
        input_embeds = original_input_embeddings.index_select(0, valid_ids_tensor).to(dtype=data_type)
        local_embedding_input = (final_weights * input_embeds).sum(dim=0)
        
        local_embedding_output = None
        if original_output_embeddings is not None:
            output_embeds = original_output_embeddings.index_select(0, valid_ids_tensor).to(dtype=data_type)
            local_embedding_output = (final_weights * output_embeds).sum(dim=0)
        
        return local_embedding_input.cpu(), local_embedding_output.cpu() if local_embedding_output is not None else None
    
    except Exception as e:
        print(f"Error calculating local embedding for token '{token_str}' (ID: {new_token_id}): {e}")
        return None, None