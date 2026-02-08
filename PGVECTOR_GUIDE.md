# pgvector Integration Guide

## Overview

pgvector 0.7.0 adds powerful vector similarity search to PostgreSQL. For FlowState, this enables:
- **Fast session similarity search** (find similar movement patterns in milliseconds)
- **Baseline comparison** (compare patient movements to ideal references)
- **Clustering** (group sessions by movement characteristics)
- **Semantic search** (find sessions by movement description)

---

## Installation

### Install pgvector Extension

```bash
# macOS
brew install pgvector

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # May require sudo
```

### Enable in Database

```bash
# In your PostgreSQL database
psql -d flowstate -c "CREATE EXTENSION vector;"
```

---

## Setup FlowState with pgvector

### 1. Initialize Vector Tables

```python
from pgvector_integration import VectorRehabDatabase

# Create database with pgvector support
db = VectorRehabDatabase()

# Setup pgvector tables and indexes
db.setup_pgvector()
```

This creates:
- `session_embeddings` table for session feature vectors
- `movement_embeddings` table for baseline movement patterns
- HNSW and IVFFlat indexes for fast similarity search

### 2. Generate Embeddings from Existing Sessions

```bash
# Create embeddings for all sessions in database
python pgvector_integration.py create
```

This will:
1. Load all completed sessions (≥50 frames)
2. Extract 22-dimensional feature vectors
3. Store in `session_embeddings` table with HNSW index
4. Enable instant similarity search

---

## Usage Examples

### Find Similar Sessions

```python
from pgvector_integration import VectorRehabDatabase

db = VectorRehabDatabase()

# Find sessions similar to a specific session
similar = db.find_similar_sessions(
    session_id='session_abc123',
    limit=10,
    min_similarity=0.7  # 0-1 scale, 1 = identical
)

for session in similar:
    print(f"Session: {session['session_id']}")
    print(f"  Similarity: {session['similarity']:.2%}")
    print(f"  Rehab Stage: {session['rehab_stage']}")
    print(f"  User: {session['user_id']}")
```

### Search with Filters

```python
# Find similar sessions from patients at same rehab stage
similar = db.find_sessions_by_criteria(
    embedding=feature_vector,
    rehab_stage='mid',
    diagnosis='stroke',
    limit=5
)
```

### Store Baseline Movement Patterns

```python
# Extract features from an "ideal" movement session
from train_models import RehabModelTrainer

trainer = RehabModelTrainer()
ideal_session = db.get_session_timeseries('ideal_session_xyz')
ideal_features = trainer.extract_features_from_session(ideal_session)

# Store as baseline
baseline_id = db.store_baseline_movement(
    movement_name='finger_extension_ideal',
    embedding=ideal_features,
    movement_type='finger_exercise',
    description='Ideal finger extension pattern for advanced rehab',
    metadata={'expert_rating': 5.0, 'source': 'therapist_demo'}
)
```

### Compare Patient to Baselines

```python
# Get patient session features
patient_session = db.get_session_timeseries('patient_session_456')
patient_features = trainer.extract_features_from_session(patient_session)

# Find matching baseline movements
baselines = db.find_baseline_movements(
    embedding=patient_features,
    movement_type='finger_exercise',
    limit=3
)

for baseline in baselines:
    print(f"Baseline: {baseline['movement_name']}")
    print(f"  Match: {baseline['similarity']:.2%}")
    print(f"  Description: {baseline['description']}")
```

---

## Integration with ML Training

### Store Model Embeddings

When training models, store the feature vectors:

```python
from train_models import RehabModelTrainer
from pgvector_integration import VectorRehabDatabase

trainer = RehabModelTrainer()
db = VectorRehabDatabase()

# During training, after extracting features
X, y, session_ids = trainer.prepare_training_data()

# Store all feature vectors
embeddings = [(session_id, features) for session_id, features in zip(session_ids, X)]
db.batch_store_embeddings(embeddings, embedding_type='feature_vector')
```

### Use for Training Data Selection

```python
# Find sessions similar to a challenging case for augmented training
challenging_session = 'session_with_high_tremor'
similar_cases = db.find_similar_sessions(
    session_id=challenging_session,
    limit=20
)

# Use these for targeted training
similar_ids = [s['session_id'] for s in similar_cases]
```

---

## Flask API Integration

### Add Similarity Search Endpoint

```python
from flask import Flask, jsonify, request
from pgvector_integration import VectorRehabDatabase
from train_models import RehabModelTrainer

app = Flask(__name__)
db = VectorRehabDatabase()
trainer = RehabModelTrainer()

@app.route('/api/session/<session_id>/similar', methods=['GET'])
def find_similar(session_id):
    """Find sessions with similar movement patterns"""
    limit = request.args.get('limit', 10, type=int)
    min_similarity = request.args.get('min_similarity', 0.5, type=float)
    
    similar = db.find_similar_sessions(
        session_id=session_id,
        limit=limit,
        min_similarity=min_similarity
    )
    
    return jsonify({
        'query_session': session_id,
        'similar_sessions': similar,
        'count': len(similar)
    })

@app.route('/api/session/<session_id>/baseline_match', methods=['GET'])
def match_baseline(session_id):
    """Compare session to baseline movement patterns"""
    # Get session features
    session_data = db.get_session_timeseries(session_id)
    features = trainer.extract_features_from_session(session_data)
    
    if features is None:
        return jsonify({'error': 'Could not extract features'}), 400
    
    # Find matching baselines
    baselines = db.find_baseline_movements(features, limit=5)
    
    return jsonify({
        'session_id': session_id,
        'matching_baselines': baselines,
        'best_match': baselines[0] if baselines else None
    })

@app.route('/api/user/<user_id>/peer_comparison', methods=['GET'])
def peer_comparison(user_id):
    """Find similar patients for peer comparison"""
    # Get user's latest session
    sessions = db.get_user_sessions(user_id, limit=1)
    
    if not sessions:
        return jsonify({'error': 'No sessions found'}), 404
    
    latest_session = sessions[0]['session_id']
    
    # Find similar sessions from other users
    similar = db.find_similar_sessions(
        session_id=latest_session,
        limit=10
    )
    
    # Filter out same user
    peer_sessions = [s for s in similar if s['user_id'] != user_id]
    
    return jsonify({
        'user_id': user_id,
        'peer_sessions': peer_sessions,
        'count': len(peer_sessions)
    })
```

---

## Performance Comparison

### pgvector vs DTW

| Feature | pgvector | DTW Analysis |
|---------|----------|--------------|
| **Speed** | <10ms for 10k sessions | 100-500ms per comparison |
| **Use Case** | Quick similarity search | Detailed alignment analysis |
| **Accuracy** | Feature-based | Sequence alignment |
| **Scale** | Millions of vectors | Hundreds of sequences |

**Recommendation**: Use pgvector for initial filtering, then DTW for detailed analysis:

```python
# 1. Fast filter with pgvector (milliseconds)
candidates = db.find_similar_sessions(session_id, limit=10)

# 2. Detailed analysis with DTW (seconds)
from dtw_analysis import MovementAnalyzer
analyzer = MovementAnalyzer()

for candidate in candidates:
    dtw_result = analyzer.compare_to_baseline(
        session_id,
        candidate['session_id']
    )
    print(f"DTW Distance: {dtw_result['dtw_distance']:.2f}")
```

---

## Advanced Features

### Binary Quantization (pgvector 0.7.0)

Reduce storage and increase speed with binary vectors:

```python
# Store binary quantized version
import numpy as np

# Original embedding
embedding = np.array([0.5, -0.3, 0.8, ...])

# Binary quantize (halfvec for 2-byte floats)
binary_embedding = (embedding > 0).astype(np.int8)

db.store_session_embedding(
    session_id='session_123',
    embedding=binary_embedding,
    embedding_type='binary_quantized'
)

# Search is 10x faster with binary vectors
similar = db.find_similar_sessions(
    session_id='session_123',
    embedding_type='binary_quantized',
    limit=100
)
```

### Sparse Vectors for Movement Events

For sparse event-based features (e.g., tremor spikes):

```python
from scipy.sparse import csr_matrix

# Create sparse vector (only store non-zero values)
sparse_features = csr_matrix([0, 0, 0.8, 0, 0, 2.1, 0, ...])

# Store (pgvector 0.7.0 supports sparsevec type)
db.store_session_embedding(
    session_id='session_123',
    embedding=sparse_features.toarray()[0],
    embedding_type='sparse_events'
)
```

### Custom Distance Functions

```python
# Use hamming distance for binary vectors
cursor.execute('''
    SELECT session_id, hamming_distance(embedding, %s) as distance
    FROM session_embeddings
    WHERE embedding_type = 'binary_quantized'
    ORDER BY distance
    LIMIT 10
''', (query_vector,))

# Use L1 distance instead of cosine
cursor.execute('''
    SELECT session_id, embedding <+> %s as l1_distance
    FROM session_embeddings
    ORDER BY embedding <+> %s
    LIMIT 10
''', (query_vector, query_vector))
```

---

## Monitoring and Stats

```python
# Get embedding statistics
stats = db.get_embedding_stats()

print("Session Embeddings:")
for emb_type in stats['session_embeddings']:
    print(f"  Type: {emb_type['embedding_type']}")
    print(f"  Count: {emb_type['count']}")
    print(f"  Dimensions: {emb_type['avg_dimension']:.0f}")
    print(f"  Date Range: {emb_type['oldest']} to {emb_type['newest']}")

print("\nMovement Baselines:")
for mov_type in stats['movement_embeddings']:
    print(f"  Type: {mov_type['movement_type']}")
    print(f"  Count: {mov_type['count']}")
```

---

## Demo

Run the included demo:

```bash
# Create embeddings for all sessions
python pgvector_integration.py create

# Demo similarity search
python pgvector_integration.py demo
```

Example output:
```
====================================================================
pgvector Similarity Search Demo
====================================================================

Query session: session_2024_01_15_abc

Found 5 similar sessions:

1. Session: session_2024_01_18_xyz
   Similarity: 0.923
   Rehab Stage: mid
   Smoothness: 0.745
   Tremor: 0.123

2. Session: session_2024_01_20_def
   Similarity: 0.891
   Rehab Stage: mid
   Smoothness: 0.738
   Tremor: 0.134
...
```

---

## Troubleshooting

### pgvector extension not found

```bash
# Verify installation
psql -d flowstate -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# Reinstall if needed
brew reinstall pgvector
psql -d flowstate -c "CREATE EXTENSION vector;"
```

### Slow queries

```python
# Check index usage
cursor.execute('''
    EXPLAIN ANALYZE
    SELECT session_id FROM session_embeddings
    ORDER BY embedding <=> %s
    LIMIT 10
''', (query_vector,))

# If not using index, rebuild:
cursor.execute('REINDEX INDEX session_embeddings_hnsw_idx;')
```

### High memory usage

```python
# Use IVFFlat instead of HNSW for larger datasets
cursor.execute('''
    CREATE INDEX session_embeddings_ivfflat_idx
    ON session_embeddings 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
''')
```

---

## References

- **pgvector GitHub**: https://github.com/pgvector/pgvector
- **pgvector 0.7.0 Release Notes**: https://github.com/pgvector/pgvector/blob/master/CHANGELOG.md#070-2024-04-29
- **Distance Functions**: https://github.com/pgvector/pgvector#distances
- **Indexing Guide**: https://github.com/pgvector/pgvector#indexing
