"""
Semantic Memory Manager
Handles vector embeddings, semantic search, and knowledge retrieval
"""

import json
import logging
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session

from ..database import get_db, MemoryFragment, SemanticMemory

logger = logging.getLogger(__name__)

class SemanticMemoryManager:
    """Manages semantic memory with vector embeddings for intelligent retrieval"""
    
    def __init__(self):
        self.db = get_db()
        self.embedding_cache = {}  # Cache for frequently used embeddings
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using production embedding service"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Try OpenAI embeddings first (if API key available)
            embedding = self._generate_openai_embedding(text)
            if embedding:
                self.embedding_cache[text] = embedding
                return embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
        
        try:
            # Fallback to sentence-transformers (local model)
            embedding = self._generate_local_embedding(text)
            if embedding:
                self.embedding_cache[text] = embedding
                return embedding
        except Exception as e:
            logger.warning(f"Local embedding failed: {e}")
        
        # Final fallback to TF-IDF based embedding
        return self._generate_tfidf_embedding(text)
    
    def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        import os
        import asyncio
        import aiohttp
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        
        try:
            async def get_embedding():
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'input': text[:8000],  # Limit text length
                    'model': 'text-embedding-3-small'  # Cost-effective model
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'https://api.openai.com/v1/embeddings',
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result['data'][0]['embedding']
                        else:
                            logger.warning(f"OpenAI API error: {response.status}")
                            return None
            
            # Run async function in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(get_embedding())
            
        except Exception as e:
            logger.warning(f"OpenAI embedding error: {e}")
            return None
    
    def _generate_local_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize model (cached after first use)
            if not hasattr(self, '_local_model'):
                # Use a lightweight model for production
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
            
            # Generate embedding
            embedding = self._local_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            logger.warning(f"Local embedding error: {e}")
            return None
    
    def _generate_tfidf_embedding(self, text: str) -> List[float]:
        """Generate TF-IDF based embedding as final fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            import re
            
            # Initialize TF-IDF vectorizer (cached)
            if not hasattr(self, '_tfidf_vectorizer'):
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self._svd = TruncatedSVD(n_components=384)  # Reduce to 384 dimensions
                
                # Initialize with some sample text if this is the first call
                sample_texts = [
                    text,
                    "trading analysis market data",
                    "cryptocurrency bitcoin ethereum price",
                    "financial investment portfolio risk"
                ]
                
                tfidf_matrix = self._tfidf_vectorizer.fit_transform(sample_texts)
                self._svd.fit(tfidf_matrix)
            
            # Transform text to TF-IDF then reduce dimensions
            tfidf_vector = self._tfidf_vectorizer.transform([text])
            reduced_vector = self._svd.transform(tfidf_vector)
            
            # Normalize
            embedding = reduced_vector[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available for TF-IDF embeddings")
            # Ultimate fallback - hash-based embedding
            return self._generate_hash_embedding(text)
        except Exception as e:
            logger.warning(f"TF-IDF embedding error: {e}")
            return self._generate_hash_embedding(text)
    
    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Generate hash-based embedding as ultimate fallback"""
        # Clean and normalize text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Create a more sophisticated hash-based embedding
        embedding = [0.0] * 384
        
        for i, word in enumerate(words[:50]):  # Limit to 50 words
            word_hash = hash(word + str(i))
            for j in range(384):
                bit = (word_hash >> (j % 64)) & 1
                embedding[j] += bit * (1.0 / (i + 1))  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return embedding
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def store_memory_fragment(self, user_id: int, fragment_type: str, content: str,
                            context: Optional[str] = None, relevance_score: float = 0.5) -> bool:
        """Store a memory fragment with embedding"""
        try:
            embedding = self.generate_embedding(content)
            
            with self.db.get_session() as session:
                fragment = MemoryFragment(
                    user_id=user_id,
                    fragment_type=fragment_type,
                    content=content,
                    context=context,
                    embedding=json.dumps(embedding),
                    relevance_score=relevance_score
                )
                session.add(fragment)
                session.commit()
                
            logger.debug(f"Stored memory fragment: {fragment_type} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory fragment: {e}")
            return False
    
    def store_memory(self, user_id: int, memory_type: str, content: str,
                    context: Optional[str] = None, keywords: Optional[List[str]] = None,
                    associated_symbols: Optional[List[str]] = None, confidence: float = 0.5) -> bool:
        """Store semantic memory"""
        try:
            embedding = self.generate_embedding(content)
            
            with self.db.get_session() as session:
                memory = SemanticMemory(
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    keywords=' '.join(keywords or []),
                    embedding=json.dumps(embedding),
                    associated_symbols=' '.join(associated_symbols or []),
                    confidence=confidence
                )
                session.add(memory)
                session.commit()
                
            logger.debug(f"Stored semantic memory: {memory_type} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}")
            return False
    
    def search_similar_memories(self, user_id: int, query: str, memory_type: Optional[str] = None,
                              limit: int = 10, min_similarity: float = 0.6) -> List[Dict]:
        """Search for similar memories using semantic similarity"""
        try:
            query_embedding = self.generate_embedding(query)
            
            with self.db.get_session() as session:
                query_filter = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id
                )
                
                if memory_type:
                    query_filter = query_filter.filter(SemanticMemory.memory_type == memory_type)
                
                memories = query_filter.all()
                
                similar_memories = []
                
                for memory in memories:
                    if memory.embedding:
                        memory_embedding = json.loads(memory.embedding)
                        similarity = self.calculate_similarity(query_embedding, memory_embedding)
                        
                        if similarity >= min_similarity:
                            similar_memories.append({
                                'id': memory.id,
                                'memory_type': memory.memory_type,
                                'content': memory.content,
                                'keywords': memory.keywords,
                                'associated_symbols': memory.associated_symbols,
                                'confidence': memory.confidence,
                                'similarity_score': similarity,
                                'reinforcement_count': memory.reinforcement_count,
                                'created_at': memory.created_at.isoformat(),
                                'last_reinforced': memory.last_reinforced.isoformat()
                            })
                
                # Sort by similarity score
                similar_memories.sort(key=lambda x: x['similarity_score'], reverse=True)
                return similar_memories[:limit]
                
        except Exception as e:
            logger.error(f"Error searching similar memories: {e}")
            return []
    
    def search_memory_fragments(self, user_id: int, query: str, fragment_type: Optional[str] = None,
                              limit: int = 10, min_similarity: float = 0.6) -> List[Dict]:
        """Search memory fragments by similarity"""
        try:
            query_embedding = self.generate_embedding(query)
            
            with self.db.get_session() as session:
                query_filter = session.query(MemoryFragment).filter(
                    MemoryFragment.user_id == user_id
                )
                
                if fragment_type:
                    query_filter = query_filter.filter(MemoryFragment.fragment_type == fragment_type)
                
                fragments = query_filter.all()
                
                similar_fragments = []
                
                for fragment in fragments:
                    if fragment.embedding:
                        fragment_embedding = json.loads(fragment.embedding)
                        similarity = self.calculate_similarity(query_embedding, fragment_embedding)
                        
                        if similarity >= min_similarity:
                            # Update access count
                            fragment.access_count += 1
                            fragment.last_accessed = datetime.utcnow()
                            
                            similar_fragments.append({
                                'id': fragment.id,
                                'fragment_type': fragment.fragment_type,
                                'content': fragment.content,
                                'context': fragment.context,
                                'relevance_score': fragment.relevance_score,
                                'similarity_score': similarity,
                                'access_count': fragment.access_count,
                                'created_at': fragment.created_at.isoformat()
                            })
                
                session.commit()
                
                # Sort by similarity score
                similar_fragments.sort(key=lambda x: x['similarity_score'], reverse=True)
                return similar_fragments[:limit]
                
        except Exception as e:
            logger.error(f"Error searching memory fragments: {e}")
            return []
    
    def get_memories_by_symbol(self, user_id: int, symbol: str, limit: int = 10) -> List[Dict]:
        """Get memories associated with a specific trading symbol"""
        try:
            with self.db.get_session() as session:
                memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id,
                    SemanticMemory.associated_symbols.like(f'%{symbol}%')
                ).order_by(SemanticMemory.last_reinforced.desc()).limit(limit).all()
                
                return [{
                    'id': memory.id,
                    'memory_type': memory.memory_type,
                    'content': memory.content,
                    'keywords': memory.keywords,
                    'confidence': memory.confidence,
                    'reinforcement_count': memory.reinforcement_count,
                    'created_at': memory.created_at.isoformat(),
                    'last_reinforced': memory.last_reinforced.isoformat()
                } for memory in memories]
                
        except Exception as e:
            logger.error(f"Error getting memories by symbol: {e}")
            return []
    
    def reinforce_memory(self, memory_id: int, confidence_boost: float = 0.1) -> bool:
        """Reinforce a memory (increase confidence and reinforcement count)"""
        try:
            with self.db.get_session() as session:
                memory = session.query(SemanticMemory).filter(
                    SemanticMemory.id == memory_id
                ).first()
                
                if memory:
                    memory.reinforcement_count += 1
                    memory.confidence = min(1.0, memory.confidence + confidence_boost)
                    memory.last_reinforced = datetime.utcnow()
                    session.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Error reinforcing memory: {e}")
            
        return False
    
    def decay_memories(self, user_id: int, decay_rate: float = 0.01) -> int:
        """Apply decay to memories that haven't been accessed recently"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=30)  # 30 days without access
            
            with self.db.get_session() as session:
                old_memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id,
                    SemanticMemory.last_reinforced < cutoff_date
                ).all()
                
                count = 0
                for memory in old_memories:
                    memory.confidence = max(0.1, memory.confidence - decay_rate)
                    count += 1
                
                session.commit()
                
            logger.info(f"Applied decay to {count} memories for user {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
            return 0
    
    def get_knowledge_graph(self, user_id: int, max_nodes: int = 50) -> Dict:
        """Build a knowledge graph from user's memories"""
        try:
            with self.db.get_session() as session:
                memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id,
                    SemanticMemory.confidence > 0.3
                ).order_by(SemanticMemory.confidence.desc()).limit(max_nodes).all()
                
                nodes = []
                edges = []
                
                for i, memory in enumerate(memories):
                    nodes.append({
                        'id': memory.id,
                        'label': memory.content[:50] + "..." if len(memory.content) > 50 else memory.content,
                        'type': memory.memory_type,
                        'confidence': memory.confidence,
                        'keywords': memory.keywords.split() if memory.keywords else []
                    })
                    
                    # Create edges based on keyword similarity
                    for j, other_memory in enumerate(memories[i+1:], i+1):
                        if memory.keywords and other_memory.keywords:
                            memory_keywords = set(memory.keywords.split())
                            other_keywords = set(other_memory.keywords.split())
                            
                            overlap = len(memory_keywords.intersection(other_keywords))
                            if overlap > 0:
                                strength = overlap / len(memory_keywords.union(other_keywords))
                                if strength > 0.2:  # Minimum connection strength
                                    edges.append({
                                        'source': memory.id,
                                        'target': other_memory.id,
                                        'strength': strength,
                                        'type': 'keyword_similarity'
                                    })
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'total_memories': len(memories),
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {'nodes': [], 'edges': [], 'total_memories': 0}
    
    def consolidate_memories(self, user_id: int, min_similarity: float = 0.8) -> int:
        """Consolidate very similar memories to reduce redundancy"""
        try:
            with self.db.get_session() as session:
                memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id
                ).all()
                
                consolidated_count = 0
                to_remove = set()
                
                for i, memory1 in enumerate(memories):
                    if memory1.id in to_remove:
                        continue
                        
                    memory1_embedding = json.loads(memory1.embedding) if memory1.embedding else []
                    
                    for j, memory2 in enumerate(memories[i+1:], i+1):
                        if memory2.id in to_remove:
                            continue
                            
                        memory2_embedding = json.loads(memory2.embedding) if memory2.embedding else []
                        
                        if memory1_embedding and memory2_embedding:
                            similarity = self.calculate_similarity(memory1_embedding, memory2_embedding)
                            
                            if similarity >= min_similarity:
                                # Merge memories - keep the one with higher confidence
                                if memory1.confidence >= memory2.confidence:
                                    # Update memory1 with combined information
                                    memory1.reinforcement_count += memory2.reinforcement_count
                                    memory1.confidence = min(1.0, memory1.confidence + 0.1)
                                    memory1.last_reinforced = datetime.utcnow()
                                    
                                    # Combine keywords
                                    keywords1 = set(memory1.keywords.split()) if memory1.keywords else set()
                                    keywords2 = set(memory2.keywords.split()) if memory2.keywords else set()
                                    memory1.keywords = ' '.join(keywords1.union(keywords2))
                                    
                                    to_remove.add(memory2.id)
                                else:
                                    # Update memory2 and mark memory1 for removal
                                    memory2.reinforcement_count += memory1.reinforcement_count
                                    memory2.confidence = min(1.0, memory2.confidence + 0.1)
                                    memory2.last_reinforced = datetime.utcnow()
                                    
                                    keywords1 = set(memory1.keywords.split()) if memory1.keywords else set()
                                    keywords2 = set(memory2.keywords.split()) if memory2.keywords else set()
                                    memory2.keywords = ' '.join(keywords1.union(keywords2))
                                    
                                    to_remove.add(memory1.id)
                                    break
                                
                                consolidated_count += 1
                
                # Remove consolidated memories
                if to_remove:
                    session.query(SemanticMemory).filter(
                        SemanticMemory.id.in_(to_remove)
                    ).delete(synchronize_session=False)
                
                session.commit()
                
            logger.info(f"Consolidated {consolidated_count} similar memories for user {user_id}")
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return 0
    
    def export_user_memories(self, user_id: int) -> Dict:
        """Export all memories for a user"""
        try:
            with self.db.get_session() as session:
                memories = session.query(SemanticMemory).filter(
                    SemanticMemory.user_id == user_id
                ).all()
                
                fragments = session.query(MemoryFragment).filter(
                    MemoryFragment.user_id == user_id
                ).all()
                
                return {
                    'user_id': user_id,
                    'semantic_memories': [{
                        'id': memory.id,
                        'memory_type': memory.memory_type,
                        'content': memory.content,
                        'keywords': memory.keywords,
                        'associated_symbols': memory.associated_symbols,
                        'confidence': memory.confidence,
                        'reinforcement_count': memory.reinforcement_count,
                        'created_at': memory.created_at.isoformat(),
                        'last_reinforced': memory.last_reinforced.isoformat()
                    } for memory in memories],
                    'memory_fragments': [{
                        'id': fragment.id,
                        'fragment_type': fragment.fragment_type,
                        'content': fragment.content,
                        'context': fragment.context,
                        'relevance_score': fragment.relevance_score,
                        'access_count': fragment.access_count,
                        'created_at': fragment.created_at.isoformat()
                    } for fragment in fragments],
                    'total_semantic_memories': len(memories),
                    'total_fragments': len(fragments),
                    'exported_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error exporting user memories: {e}")
            return {}