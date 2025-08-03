import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from xinference.client import Client
import jsonlines
from collections import defaultdict
from datetime import datetime

# ------------------------------
# Constant definitions
# ------------------------------
DEFAULT_VECTOR_DIM = 768  # Default vector dimension
DEFAULT_BATCH_SIZE = 32  # Default batch processing size
MAX_SEARCH_K = 200  # Maximum search neighbor count
MAX_DAYS_DIFF = 30  # Maximum day difference for date similarity calculation

# Similarity weight configuration
WEIGHT_FAISS = 1  # FAISS vector similarity weight
WEIGHT_SUBJECT = 0  # Subject similarity weight
WEIGHT_LOCATION = 0  # Location similarity weight
WEIGHT_TAGS = 0  # Tags similarity weight
WEIGHT_DATE = 0  # Date similarity weight
TOTAL_WEIGHT = WEIGHT_FAISS + WEIGHT_SUBJECT + WEIGHT_LOCATION + WEIGHT_TAGS + WEIGHT_DATE


class TextVectorizer:
    """Tool class for obtaining text embedding vectors using Xinference

    This class encapsulates interaction with Xinference service, providing text-to-vector conversion functionality,
    supporting single text and batch text vector generation.
    """

    def __init__(self, xinference_url="", model_name="bge-m3"): #Put embedding model url here, this file uses xinference as example, can be replaced with others
        """Initialize Xinference client

        Args:
            xinference_url (str): Xinference service address
            model_name (str): Model name for generating embeddings
        """
        self.client = Client(xinference_url)
        self.model = self.client.get_model(model_name)

    def get_embedding(self, description):
        """Get embedding vector for a single description

        Args:
            description (str): Text description to be converted to vector

        Returns:
            list: Text embedding vector, returns zero vector if error occurs
        """
        try:
            response = self.model.create_embedding(description)
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Failed to get description embedding: {description[:50]}... Error: {e}")
            return [0.0] * DEFAULT_VECTOR_DIM

    def batch_get_embeddings(self, descriptions, batch_size=DEFAULT_BATCH_SIZE):
        """Batch get description embedding vectors, supports progress bar display

        Args:
            descriptions (list): List of text descriptions
            batch_size (int): Batch processing size

        Returns:
            np.array: Embedding vector array, shape (len(descriptions), vector_dim)
        """
        embeddings = []
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Generating vectors"):
            batch_descriptions = descriptions[i:i + batch_size]
            batch_embeddings = [self.get_embedding(desc) for desc in batch_descriptions]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings, dtype=np.float32)


class VectorDatabase:
    """Database class using FAISS to manage vectors and save to local

    This class encapsulates FAISS vector index creation, addition, search and persistence functionality,
    while maintaining ID to description and metadata mapping relationships.
    """

    def __init__(self, dim=DEFAULT_VECTOR_DIM, index_type="Flat", metric=faiss.METRIC_INNER_PRODUCT):
        """Initialize FAISS index

        Args:
            dim (int): Vector dimension
            index_type (str): Index type, currently only supports "Flat"
            metric (int): Distance metric method, FAISS.METRIC_INNER_PRODUCT or FAISS.METRIC_L2
        """
        if index_type == "Flat":
            # Exact search index
            self.index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        else:
            # Default to inner product Flat index
            self.index = faiss.IndexFlatIP(dim)

        self.dim = dim
        self.ids_to_description = {}  # Store ID to description mapping
        self.ids_to_metadata = {}  # Store ID to metadata mapping

    def add_vectors(self, vectors, descriptions, metadata_list=None):
        """Add vectors to FAISS index and save mapping relationships"""
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(descriptions))]

        # Assign unique ID to each vector
        ids = np.arange(len(self.ids_to_description), len(self.ids_to_description) + len(descriptions))

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Save ID to description and metadata mapping
        for i, (desc, metadata) in enumerate(zip(descriptions, metadata_list)):
            self.ids_to_description[ids[i]] = desc
            self.ids_to_metadata[ids[i]] = metadata

        return ids

    def search(self, query_vector, k=10):
        """Search similar vectors in FAISS index"""
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means not found
                results.append({
                    "id": idx,
                    "description": self.ids_to_description[idx],
                    "metadata": self.ids_to_metadata[idx],
                    "distance": distances[0][i]
                })
        return results

    def save(self, index_path, mapping_path):
        """Save FAISS index and mapping to local files"""
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save mapping
        mapping_data = {
            "ids_to_description": self.ids_to_description,
            "ids_to_metadata": self.ids_to_metadata,
            "dim": self.dim
        }
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path, mapping_path):
        """Load FAISS index and mapping from local files"""
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.index = index
        instance.ids_to_description = mapping_data["ids_to_description"]
        instance.ids_to_metadata = mapping_data["ids_to_metadata"]
        instance.dim = mapping_data["dim"]
        
        return instance


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")


def extract_descriptions_and_metadata(data):
    """Extract descriptions and metadata from data"""
    descriptions = []
    metadata_list = []
    
    for item in data:
        # Extract description (text content)
        description = item.get('description', '')
        if not description:
            description = item.get('text', '')
        if not description:
            description = item.get('content', '')
        
        # Extract metadata
        metadata = {
            'id': item.get('id', ''),
            'subject': item.get('subject', ''),
            'location': item.get('location', ''),
            'tags': item.get('tags', []),
            'date': item.get('date', ''),
            'source': item.get('source', ''),
            'incident_id': item.get('incident_id', '')
        }
        
        descriptions.append(description)
        metadata_list.append(metadata)
    
    return descriptions, metadata_list


def process_data(input_file, output_dir, batch_size=DEFAULT_BATCH_SIZE, max_items=None):
    """Process data and generate vectors"""
    print(f"Processing file: {input_file}")
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    if max_items:
        data = data[:max_items]
        print(f"Limited to {max_items} items")
    
    print(f"Total items: {len(data)}")
    
    # Extract descriptions and metadata
    descriptions, metadata_list = extract_descriptions_and_metadata(data)
    
    # Initialize vectorizer
    vectorizer = TextVectorizer()
    
    # Generate vectors
    vectors = vectorizer.batch_get_embeddings(descriptions, batch_size)
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Add vectors to database
    vector_db.add_vectors(vectors, descriptions, metadata_list)
    
    # Save to files
    index_path = os.path.join(output_dir, "faiss_index.faiss")
    mapping_path = os.path.join(output_dir, "vector_mapping.json")
    vector_db.save(index_path, mapping_path)
    
    print(f"Saved vectors to: {output_dir}")
    return vector_db


def calculate_date_similarity(date1_str, date2_str, max_days_diff=MAX_DAYS_DIFF):
    """Calculate date similarity"""
    try:
        if not date1_str or not date2_str:
            return 0.0
        
        # Parse dates
        date1 = datetime.strptime(date1_str, "%Y-%m-%d")
        date2 = datetime.strptime(date2_str, "%Y-%m-%d")
        
        # Calculate day difference
        days_diff = abs((date1 - date2).days)
        
        # Convert to similarity (0-1)
        if days_diff <= max_days_diff:
            similarity = 1.0 - (days_diff / max_days_diff)
        else:
            similarity = 0.0
        
        return similarity
    except:
        return 0.0


def calculate_jaccard_similarity(list1, list2):
    """Calculate Jaccard similarity between two lists"""
    if not list1 and not list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_exact_match_similarity(str1, str2):
    """Calculate exact match similarity between two strings"""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    return 1.0 if str1.lower() == str2.lower() else 0.0


def calculate_fuzzy_similarity(str1, str2):
    """Calculate fuzzy similarity between two strings using Levenshtein distance"""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(str1, str2)
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def build_event_clusters_from_faiss(vector_db, high_threshold=0.78):
    """Build event clusters from FAISS search results"""
    print("Building event clusters...")
    
    # Get all descriptions
    all_descriptions = list(vector_db.ids_to_description.values())
    all_metadata = list(vector_db.ids_to_metadata.values())
    
    # Track processed items
    processed = set()
    clusters = []
    
    for i, (description, metadata) in enumerate(zip(all_descriptions, all_metadata)):
        if i in processed:
            continue
        
        # Search for similar items
        query_vector = vector_db.index.reconstruct(i)
        search_results = vector_db.search(query_vector, k=MAX_SEARCH_K)
        
        # Build cluster
        cluster_items = []
        cluster_ids = set()
        
        for result in search_results:
            result_id = result["id"]
            if result_id in processed:
                continue
            
            # Calculate comprehensive similarity
            result_metadata = vector_db.ids_to_metadata[result_id]
            
            # FAISS similarity
            faiss_sim = 1.0 - result["distance"] if result["distance"] <= 1.0 else 0.0
            
            # Metadata similarities
            subject_sim = calculate_exact_match_similarity(
                metadata.get('subject', ''), 
                result_metadata.get('subject', '')
            )
            
            location_sim = calculate_exact_match_similarity(
                metadata.get('location', ''), 
                result_metadata.get('location', '')
            )
            
            tags_sim = calculate_jaccard_similarity(
                metadata.get('tags', []), 
                result_metadata.get('tags', [])
            )
            
            date_sim = calculate_date_similarity(
                metadata.get('date', ''), 
                result_metadata.get('date', '')
            )
            
            # Weighted similarity
            weighted_sim = (
                WEIGHT_FAISS * faiss_sim +
                WEIGHT_SUBJECT * subject_sim +
                WEIGHT_LOCATION * location_sim +
                WEIGHT_TAGS * tags_sim +
                WEIGHT_DATE * date_sim
            ) / TOTAL_WEIGHT
            
            # Add to cluster if similarity is high enough
            if weighted_sim >= high_threshold:
                cluster_items.append({
                    "id": result_id,
                    "description": result["description"],
                    "metadata": result_metadata,
                    "similarity": weighted_sim
                })
                cluster_ids.add(result_id)
        
        # Create cluster if we have items
        if cluster_items:
            # Sort by similarity
            cluster_items.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Create cluster
            cluster = {
                "incident_id": f"cluster_{len(clusters)}",
                "cases": [item["id"] for item in cluster_items],
                "descriptions": [item["description"] for item in cluster_items],
                "metadata": [item["metadata"] for item in cluster_items],
                "similarities": [item["similarity"] for item in cluster_items],
                "cluster_size": len(cluster_items)
            }
            
            clusters.append(cluster)
            
            # Mark as processed
            processed.update(cluster_ids)
            
            print(f"Created cluster {len(clusters)} with {len(cluster_items)} items")
    
    print(f"Total clusters created: {len(clusters)}")
    return clusters


def main():
    """Main function"""
    # Configuration
    input_file = "data/dataset.jsonl"
    output_dir = "event_alignment_text"
    batch_size = DEFAULT_BATCH_SIZE
    max_items = None  # Set to None to process all items
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Process data
    vector_db = process_data(input_file, output_dir, batch_size, max_items)
    
    # Build clusters
    clusters = build_event_clusters_from_faiss(vector_db)
    
    # Save clusters
    output_file = os.path.join(output_dir, "event_alignment_text.jsonl")
    with jsonlines.open(output_file, 'w') as writer:
        for cluster in clusters:
            writer.write(cluster)
    
    print(f"Saved clusters to: {output_file}")
    print(f"Total clusters: {len(clusters)}")


def calculate_fuzzy_similarity(str1, str2):
    """Calculate fuzzy similarity between two strings using Levenshtein distance"""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(str1, str2)
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


if __name__ == "__main__":
    main()