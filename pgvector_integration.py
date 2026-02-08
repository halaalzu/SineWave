"""
pgvector Integration for FlowState
Fast vector similarity search for movement patterns and sessions
"""

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import List, Dict, Tuple, Optional

try:
    from database_postgres import PostgresRehabDatabase
except ImportError:
    print("WARNING: PostgreSQL not available")


class VectorRehabDatabase(PostgresRehabDatabase):
    """
    Extended PostgreSQL database with pgvector support
    Enables fast similarity search for movement patterns
    """
    
    def setup_pgvector(self):
        """
        Enable pgvector extension and create vector tables
        
        Installation:
            brew install pgvector
            # or for manual install:
            git clone https://github.com/pgvector/pgvector.git
            cd pgvector
            make
            make install
        """
        cursor = self.conn.cursor()
        
        # Enable pgvector extension
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            print("✓ pgvector extension enabled")
        except Exception as e:
            print(f"❌ Failed to enable pgvector: {e}")
            print("Install with: brew install pgvector")
            self.conn.rollback()
            return False
        
        # Create session embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_embeddings (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL REFERENCES sessions(session_id),
                embedding_type VARCHAR(100) NOT NULL,
                vector_dimension INTEGER NOT NULL,
                embedding vector NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                UNIQUE(session_id, embedding_type)
            )
        ''')
        
        # Create HNSW index for fast similarity search
        # HNSW (Hierarchical Navigable Small World) is faster than IVFFlat for most cases
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS session_embeddings_hnsw_idx
            ON session_embeddings 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        ''')
        
        # Also create IVFFlat index option (better for larger datasets)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS session_embeddings_ivfflat_idx
            ON session_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        ''')
        
        # Create movement pattern embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_embeddings (
                id SERIAL PRIMARY KEY,
                movement_name VARCHAR(255) NOT NULL,
                movement_type VARCHAR(100),
                embedding vector NOT NULL,
                reference_session_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                metadata JSONB
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS movement_embeddings_hnsw_idx
            ON movement_embeddings 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        ''')
        
        self.conn.commit()
        print("✓ pgvector tables and indexes created")
        return True
    
    def store_session_embedding(self, session_id: str, embedding: np.ndarray,
                               embedding_type: str = 'feature_vector',
                               metadata: Dict = None):
        """
        Store feature vector for a session
        
        Args:
            session_id: Session identifier
            embedding: Feature vector (numpy array)
            embedding_type: Type of embedding ('feature_vector', 'lstm_hidden', 'autoencoder')
            metadata: Additional metadata (model version, etc.)
        """
        cursor = self.conn.cursor()
        
        # Convert numpy array to list for PostgreSQL
        vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        vector_dim = len(vector_list)
        
        cursor.execute('''
            INSERT INTO session_embeddings 
            (session_id, embedding_type, vector_dimension, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id, embedding_type) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                vector_dimension = EXCLUDED.vector_dimension,
                metadata = EXCLUDED.metadata,
                created_at = CURRENT_TIMESTAMP
        ''', (session_id, embedding_type, vector_dim, vector_list, json.dumps(metadata or {})))
        
        self.conn.commit()
    
    def find_similar_sessions(self, session_id: str = None, 
                             embedding: np.ndarray = None,
                             limit: int = 10,
                             embedding_type: str = 'feature_vector',
                             min_similarity: float = 0.0) -> List[Dict]:
        """
        Find sessions similar to a given session or embedding
        
        Args:
            session_id: Find sessions similar to this session
            embedding: Or provide embedding directly
            limit: Number of results
            embedding_type: Type of embedding to search
            min_similarity: Minimum cosine similarity (0-1)
        
        Returns:
            List of similar sessions with similarity scores
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        if session_id:
            # Get embedding for the session
            cursor.execute('''
                SELECT embedding FROM session_embeddings
                WHERE session_id = %s AND embedding_type = %s
            ''', (session_id, embedding_type))
            
            result = cursor.fetchone()
            if not result:
                return []
            
            query_vector = result['embedding']
        elif embedding is not None:
            query_vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        else:
            raise ValueError("Must provide either session_id or embedding")
        
        # Find similar sessions using cosine similarity
        # 1 - cosine_distance gives cosine similarity
        cursor.execute('''
            SELECT 
                se.session_id,
                se.embedding_type,
                1 - (se.embedding <=> %s::vector) as similarity,
                s.user_id,
                s.level_name,
                s.start_time,
                s.duration_seconds,
                s.avg_smoothness,
                s.avg_tremor,
                s.avg_speed,
                u.rehab_stage,
                u.diagnosis
            FROM session_embeddings se
            JOIN sessions s ON se.session_id = s.session_id
            JOIN users u ON s.user_id = u.user_id
            WHERE se.embedding_type = %s
                AND se.session_id != %s
                AND (1 - (se.embedding <=> %s::vector)) >= %s
            ORDER BY se.embedding <=> %s::vector
            LIMIT %s
        ''', (query_vector, embedding_type, session_id or '', 
              query_vector, min_similarity, query_vector, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def find_sessions_by_criteria(self, embedding: np.ndarray,
                                 rehab_stage: str = None,
                                 diagnosis: str = None,
                                 limit: int = 10) -> List[Dict]:
        """
        Find similar sessions with additional filters
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        where_clauses = []
        params = [vector_list, vector_list]
        
        if rehab_stage:
            where_clauses.append("u.rehab_stage = %s")
            params.append(rehab_stage)
        
        if diagnosis:
            where_clauses.append("u.diagnosis LIKE %s")
            params.append(f"%{diagnosis}%")
        
        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
        
        params.append(limit)
        
        cursor.execute(f'''
            SELECT 
                se.session_id,
                1 - (se.embedding <=> %s::vector) as similarity,
                s.*, u.rehab_stage, u.diagnosis
            FROM session_embeddings se
            JOIN sessions s ON se.session_id = s.session_id
            JOIN users u ON s.user_id = u.user_id
            WHERE se.embedding_type = 'feature_vector'
                {where_sql}
            ORDER BY se.embedding <=> %s::vector
            LIMIT %s
        ''', params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def store_baseline_movement(self, movement_name: str, 
                               embedding: np.ndarray,
                               movement_type: str = None,
                               reference_session_id: str = None,
                               description: str = None,
                               metadata: Dict = None):
        """
        Store a baseline/ideal movement pattern
        """
        cursor = self.conn.cursor()
        
        vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        cursor.execute('''
            INSERT INTO movement_embeddings
            (movement_name, movement_type, embedding, reference_session_id, description, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (movement_name, movement_type, vector_list, 
              reference_session_id, description, json.dumps(metadata or {})))
        
        movement_id = cursor.fetchone()[0]
        self.conn.commit()
        return movement_id
    
    def find_baseline_movements(self, embedding: np.ndarray,
                               movement_type: str = None,
                               limit: int = 5) -> List[Dict]:
        """
        Find baseline movements similar to given embedding
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        where_clause = "WHERE movement_type = %s" if movement_type else ""
        params = [vector_list, vector_list]
        
        if movement_type:
            params.append(movement_type)
        
        params.append(limit)
        
        cursor.execute(f'''
            SELECT 
                id,
                movement_name,
                movement_type,
                1 - (embedding <=> %s::vector) as similarity,
                reference_session_id,
                description,
                metadata
            FROM movement_embeddings
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ''', params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def batch_store_embeddings(self, embeddings: List[Tuple[str, np.ndarray]],
                              embedding_type: str = 'feature_vector'):
        """
        Store multiple session embeddings efficiently
        
        Args:
            embeddings: List of (session_id, embedding) tuples
            embedding_type: Type of embedding
        """
        cursor = self.conn.cursor()
        
        data = []
        for session_id, embedding in embeddings:
            vector_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            vector_dim = len(vector_list)
            data.append((session_id, embedding_type, vector_dim, vector_list, '{}'))
        
        from psycopg2.extras import execute_batch
        
        execute_batch(cursor, '''
            INSERT INTO session_embeddings 
            (session_id, embedding_type, vector_dimension, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id, embedding_type) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                vector_dimension = EXCLUDED.vector_dimension
        ''', data, page_size=100)
        
        self.conn.commit()
        print(f"✓ Stored {len(embeddings)} embeddings")
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT 
                embedding_type,
                COUNT(*) as count,
                AVG(vector_dimension) as avg_dimension,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM session_embeddings
            GROUP BY embedding_type
        ''')
        
        stats = {
            'session_embeddings': [dict(row) for row in cursor.fetchall()]
        }
        
        cursor.execute('''
            SELECT 
                movement_type,
                COUNT(*) as count
            FROM movement_embeddings
            GROUP BY movement_type
        ''')
        
        stats['movement_embeddings'] = [dict(row) for row in cursor.fetchall()]
        
        return stats


def create_embeddings_from_ml_features(db_path='flowstate.db'):
    """
    Generate and store embeddings for all existing sessions
    Uses ML feature extraction
    """
    from train_models import RehabModelTrainer
    
    print("\n" + "="*60)
    print("Creating embeddings from existing sessions")
    print("="*60 + "\n")
    
    # Initialize
    vector_db = VectorRehabDatabase()
    
    # Setup pgvector
    if not vector_db.setup_pgvector():
        print("❌ Failed to setup pgvector")
        return
    
    trainer = RehabModelTrainer()
    
    # Get all sessions
    sessions = vector_db.get_all_sessions_for_training(min_frames=50)
    print(f"Found {len(sessions)} sessions to process")
    
    embeddings = []
    
    for i, session in enumerate(sessions):
        session_id = session['session_id']
        
        # Get time series data
        timeseries = vector_db.get_session_timeseries(session_id)
        
        if len(timeseries) < 50:
            continue
        
        # Extract features (this becomes our embedding)
        features = trainer.extract_features_from_session(timeseries)
        
        if features is not None:
            embeddings.append((session_id, features))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sessions)} sessions")
    
    # Batch store embeddings
    if embeddings:
        vector_db.batch_store_embeddings(embeddings, embedding_type='feature_vector')
        print(f"\n✓ Created {len(embeddings)} embeddings")
        
        # Show stats
        stats = vector_db.get_embedding_stats()
        print(f"\nEmbedding Statistics:")
        for emb_type in stats['session_embeddings']:
            print(f"  Type: {emb_type['embedding_type']}")
            print(f"  Count: {emb_type['count']}")
            print(f"  Dimensions: {emb_type['avg_dimension']:.0f}")
    
    vector_db.close()


def demo_similarity_search():
    """
    Demo: Find similar sessions using vector search
    """
    print("\n" + "="*60)
    print("pgvector Similarity Search Demo")
    print("="*60 + "\n")
    
    db = VectorRehabDatabase()
    
    # Get a random session
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT session_id FROM session_embeddings 
        ORDER BY RANDOM() 
        LIMIT 1
    ''')
    
    result = cursor.fetchone()
    if not result:
        print("No embeddings found. Run create_embeddings_from_ml_features() first.")
        return
    
    query_session = result[0]
    print(f"Query session: {query_session}")
    
    # Find similar sessions
    similar = db.find_similar_sessions(
        session_id=query_session,
        limit=5,
        min_similarity=0.5
    )
    
    print(f"\nFound {len(similar)} similar sessions:\n")
    
    for i, session in enumerate(similar, 1):
        print(f"{i}. Session: {session['session_id']}")
        print(f"   Similarity: {session['similarity']:.3f}")
        print(f"   Rehab Stage: {session['rehab_stage']}")
        print(f"   Smoothness: {session['avg_smoothness']:.3f}")
        print(f"   Tremor: {session['avg_tremor']:.3f}")
        print()
    
    db.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'create':
        # Create embeddings for all sessions
        create_embeddings_from_ml_features()
    elif len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Demo similarity search
        demo_similarity_search()
    else:
        print("Usage:")
        print("  python pgvector_integration.py create  - Create embeddings")
        print("  python pgvector_integration.py demo    - Demo similarity search")
