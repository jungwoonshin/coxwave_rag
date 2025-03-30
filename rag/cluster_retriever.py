
import logging
import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ClusteringManager:
    """클러스터링 및 관련 메타데이터를 관리하는 클래스"""
    
    def __init__(self, cache_dir: str = "clustering_cache"):
        """
        클러스터링 관리자 초기화
        
        Args:
            cache_dir: 클러스터링 캐시 디렉토리 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.clusters = None
        self.cluster_centers = None
        self.id_to_cluster = {}  # 문서 ID -> 클러스터 ID 매핑
        self.cluster_to_ids = {}  # 클러스터 ID -> 문서 ID 리스트 매핑
    
    def create_clusters(self, embeddings: List[List[float]], ids: List[str], n_clusters: int = 50) -> None:
        """
        임베딩 벡터를 클러스터링하고 메타데이터 생성
        
        Args:
            embeddings: 임베딩 벡터 리스트
            ids: 임베딩에 대응하는 문서 ID 리스트
            n_clusters: 클러스터 수 (기본값: 50)
        """
        # 데이터 크기에 따라 클러스터 수 조정
        n_clusters = min(n_clusters, len(embeddings) // 10)
        n_clusters = max(n_clusters, 5)  # 최소 5개는 유지
        
        logger.info(f"Creating {n_clusters} clusters from {len(embeddings)} documents")
        
        # NumPy 배열로 변환
        embeddings_array = np.array(embeddings)
        
        # K-means 클러스터링 수행
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # 클러스터 중심점 저장
        self.cluster_centers = kmeans.cluster_centers_.tolist()
        self.clusters = cluster_labels.tolist()
        
        # 매핑 구성
        for i, (doc_id, cluster_id) in enumerate(zip(ids, cluster_labels)):
            self.id_to_cluster[doc_id] = int(cluster_id)
            
            if cluster_id not in self.cluster_to_ids:
                self.cluster_to_ids[int(cluster_id)] = []
            
            self.cluster_to_ids[int(cluster_id)].append(doc_id)
        
        # 클러스터 통계 출력
        cluster_sizes = [len(ids) for ids in self.cluster_to_ids.values()]
        logger.info(f"Cluster statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")
        
        # 캐시 저장
        self.save_clusters()
    
    def save_clusters(self) -> None:
        """클러스터링 정보를 파일에 저장"""
        try:
            cache_data = {
                "cluster_centers": self.cluster_centers,
                "id_to_cluster": self.id_to_cluster,
                "cluster_to_ids": self.cluster_to_ids
            }
            
            with open(self.cache_dir / "clusters.json", "w") as f:
                json.dump(cache_data, f)
                
            logger.info(f"Saved clustering data to {self.cache_dir / 'clusters.json'}")
        except Exception as e:
            logger.error(f"Error saving clustering data: {e}")
    
    def load_clusters(self) -> bool:
        """
        저장된 클러스터링 정보 로드
        
        Returns:
            성공 여부
        """
        cache_file = self.cache_dir / "clusters.json"
        if not cache_file.exists():
            logger.info(f"No clustering cache found at {cache_file}")
            return False
            
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                
            self.cluster_centers = cache_data["cluster_centers"]
            self.id_to_cluster = {k: int(v) for k, v in cache_data["id_to_cluster"].items()}
            self.cluster_to_ids = {int(k): v for k, v in cache_data["cluster_to_ids"].items()}
            
            logger.info(f"Loaded clustering data with {len(self.cluster_centers)} clusters")
            return True
        except Exception as e:
            logger.error(f"Error loading clustering data: {e}")
            return False
    
    def get_nearest_cluster(self, query_vector: List[float]) -> Tuple[int, float]:
        """
        쿼리 벡터에 가장 가까운 클러스터 찾기
        
        Args:
            query_vector: 쿼리 임베딩 벡터
            
        Returns:
            (클러스터 ID, 유사도 점수) 튜플
        """
        if not self.cluster_centers:
            raise ValueError("Clusters have not been created or loaded")
            
        # 벡터 변환
        query_np = np.array(query_vector)
        centers_np = np.array(self.cluster_centers)
        
        # 코사인 유사도 계산
        norm_query = np.linalg.norm(query_np)
        norm_centers = np.linalg.norm(centers_np, axis=1)
        
        dot_product = np.dot(centers_np, query_np)
        similarities = dot_product / (norm_centers * norm_query)
        
        # 가장 유사한 클러스터 찾기
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return int(best_idx), float(best_score)
    
    def get_cluster_ids(self, cluster_id: int) -> List[str]:
        """
        특정 클러스터에 속한 문서 ID 목록 반환
        
        Args:
            cluster_id: 클러스터 ID
            
        Returns:
            해당 클러스터의 문서 ID 리스트
        """
        return self.cluster_to_ids.get(cluster_id, [])


class ClusteredChromaRetriever:
    """
    클러스터링을 활용한 ChromaDB 기반 벡터 검색 클래스
    """
    def __init__(
        self,
        embedder=None,
        collection_name: str = "document_store",
        db_path: str = "chroma_data",
        cache_dir: str = "embedding_cache",
        cluster_cache_dir: str = "clustering_cache",
        embedding_dimension: int = 1536,
        top_k: int = 10,
        metric_type: str = "cosine",
        n_clusters: int = 50,
        use_clustering: bool = True
    ):
        """
        클러스터링 기반 ChromaRetriever 초기화
        
        Args:
            embedder: 텍스트 임베딩 모델 인스턴스
            collection_name: 사용할 컬렉션 이름
            db_path: ChromaDB 데이터 디렉토리 경로
            cache_dir: 임베딩 캐시 디렉토리 경로
            cluster_cache_dir: 클러스터링 캐시 디렉토리 경로
            embedding_dimension: 임베딩 차원
            top_k: 기본 반환 결과 수
            metric_type: 유사도 메트릭 (cosine, euclidean, dot_product)
            n_clusters: 클러스터 수
            use_clustering: 클러스터링 사용 여부
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_dimension = embedding_dimension
        self.metric_type = metric_type
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(exist_ok=True)
        
        # 클러스터링 관리자 초기화
        self.clustering = ClusteringManager(cluster_cache_dir)
        
        try:
            # ChromaDB 디렉토리 생성
            os.makedirs(db_path, exist_ok=True)
            
            # ChromaDB 클라이언트 초기화
            logger.info(f"Connecting to ChromaDB at {db_path}")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # 캐시 초기화
            self._init_cache()
            
            # 컬렉션 초기화
            self._ensure_collection_exists()
            
            # 클러스터링 데이터 로드 시도
            if self.use_clustering:
                self.clustering.load_clusters()
            
        except Exception as e:
            logger.error(f"Error initializing ClusteredChromaRetriever: {e}")
            raise
    
    def _init_cache(self):
        """임베딩 캐시 시스템 초기화"""
        # 메타데이터 파일이 없으면 생성
        metadata_path = self.cache_dir / "metadata.json"
        self.cache_metadata = {}
        self.model_name = "unknown"
        
        if metadata_path.exists():
            try:
                # 기존 메타데이터 로드
                with open(metadata_path, "r") as f:
                    self.cache_metadata = json.load(f)
                self.model_name = self.cache_metadata.get("model", "unknown")
                logger.info(f"Using cache for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                
            # 모델 변경 확인
            if self.embedder and self.model_name != self.embedder.model_name:
                logger.warning(f"Embedding model mismatch: cache uses {self.model_name}, current is {self.embedder.model_name}")
                self.model_name = self.embedder.model_name
        else:
            # 새 메타데이터 생성
            if self.embedder:
                self.model_name = self.embedder.model_name
            with open(metadata_path, "w") as f:
                json.dump({"model": self.model_name}, f)
            logger.info(f"Created new cache metadata for model: {self.model_name}")
        
        # 모델별 캐시 로드
        self.embedding_cache = {}
        self.cache_file = self.cache_dir / f"{self.model_name}_embeddings.json"
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Error loading embeddings cache: {e}")
                self.embedding_cache = {}
        else:
            logger.info(f"No existing cache file found at {self.cache_file}, will create as needed")
    
    def _get_embedding_with_cache(self, text: str) -> List[float]:
        """
        캐싱을 활용한 텍스트 임베딩 가져오기
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        if not self.embedder:
            raise ValueError("No embedder provided for generating embeddings")
        
        # 텍스트 해시를 캐시 키로 사용
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 캐시에 임베딩이 있는지 확인
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # 새 임베딩 생성
        vector = self.embedder.get_embedding(text)
        
        # 캐시 업데이트
        self.embedding_cache[text_hash] = vector
        
        # 주기적으로 캐시 저장 (10개 새 임베딩마다)
        if len(self.embedding_cache) % 10 == 0:
            try:
                with open(self.cache_file, "w") as f:
                    json.dump(self.embedding_cache, f)
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
        
        return vector
    
    def _ensure_collection_exists(self):
        """컬렉션이 없으면 생성"""
        try:
            # 컬렉션이 있는지 확인 또는 생성
            try:
                # 먼저 컬렉션 가져오기 시도
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except:
                # 없으면 컬렉션 생성
                logger.info(f"Creating collection: {self.collection_name}")
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.metric_type}  # 거리 메트릭 설정
                )
                
        except Exception as e:
            logger.error(f"Error creating/getting collection: {e}")
            raise
    
    def setup_collection(self, qa_pairs: List[Dict[str, Any]]) -> None:
        """
        FAQ 데이터로 컬렉션 설정
        
        Args:
            qa_pairs: 임베딩이 있거나 없는 질문-답변 쌍 리스트
        """
        try:
            # 기존 컬렉션이 있으면 삭제
            try:
                self.chroma_client.delete_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            except:
                logger.info(f"No existing collection to drop")
            
            # 컬렉션 재생성
            self._ensure_collection_exists()
            
            # 삽입할 데이터 준비
            ids = []
            questions = []
            answers = []
            embeddings = []
            
            for i, pair in enumerate(qa_pairs):
                # 임베딩 가져오기 또는 생성
                if "vector" in pair and pair["vector"]:
                    # 제공된 벡터 사용 (initialize_db.py에서)
                    vector = pair["vector"]
                    
                    # 이 벡터로 캐시 업데이트
                    question = pair.get("question", "")
                    text_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
                    self.embedding_cache[text_hash] = vector
                elif "question" in pair:
                    # 캐시된 임베딩 생성 사용
                    vector = self._get_embedding_with_cache(pair["question"])
                else:
                    logger.warning(f"Skipping pair {i}: No vector or question")
                    continue
                
                # ID 가져오기 또는 인덱스 사용
                doc_id = str(pair.get("id", i))  # ChromaDB는 문자열 ID 필요
                
                # 배치에 추가
                ids.append(doc_id)
                questions.append(pair.get("question", ""))
                answers.append(pair.get("answer", ""))
                embeddings.append(vector)
                
                # 주기적으로 진행 상황 로깅
                if (i + 1) % 100 == 0:
                    logger.info(f"Prepared {i + 1} documents for insertion")
            
            # 업데이트된 캐시 저장
            with open(self.cache_file, "w") as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            
            # 클러스터링 수행 (사용 설정된 경우)
            if self.use_clustering:
                self.clustering.create_clusters(embeddings, ids, n_clusters=self.n_clusters)
            
            # 배치로 삽입 (ChromaDB는 배치 크기 제한 있음)
            BATCH_SIZE = 500
            for i in range(0, len(ids), BATCH_SIZE):
                batch_ids = ids[i:i+BATCH_SIZE]
                batch_questions = questions[i:i+BATCH_SIZE]
                batch_answers = answers[i:i+BATCH_SIZE]
                batch_embeddings = embeddings[i:i+BATCH_SIZE]
                
                # 클러스터 정보를 메타데이터에 추가
                metadata_batch = []
                for j, (doc_id, answer) in enumerate(zip(batch_ids, batch_answers)):
                    metadata = {"answer": answer}
                    
                    # 클러스터링 사용 시 클러스터 ID 추가
                    if self.use_clustering and doc_id in self.clustering.id_to_cluster:
                        metadata["cluster_id"] = self.clustering.id_to_cluster[doc_id]
                    
                    metadata_batch.append(metadata)
                
                # 컬렉션에 임베딩 추가
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_questions,
                    metadatas=metadata_batch
                )
                
                logger.info(f"Inserted batch of {len(batch_ids)} documents")
                
            logger.info(f"Inserted total of {len(ids)} documents into collection {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def retrieve(self, query: str = None, query_vector: List[float] = None, top_k: Optional[int] = None, 
                use_clustering: Optional[bool] = None) -> List[Dict]:
        """
        벡터 검색을 통한 유사 문서 검색
        
        Args:
            query: 쿼리 텍스트 (query_vector가 제공되지 않은 경우 사용)
            query_vector: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수 (기본값: self.top_k)
            use_clustering: 클러스터링 사용 여부 (None이면 인스턴스 설정 사용)
            
        Returns:
            메타데이터와 점수가 포함된 유사 문서 리스트
        """
        if top_k is None:
            top_k = self.top_k
        
        if use_clustering is None:
            use_clustering = self.use_clustering
            
        # 필요한 경우 텍스트에서 쿼리 벡터 가져오기
        if query_vector is None:
            if query is None:
                raise ValueError("query_vector 또는 query 중 하나는 제공되어야 합니다")
            
            # 쿼리에 캐시된 임베딩 사용
            query_vector = self._get_embedding_with_cache(query)
        
        try:
            # 클러스터링 기반 검색
            if use_clustering and self.clustering.cluster_centers:
                # 가장 가까운 클러스터 찾기
                nearest_cluster, cluster_score = self.clustering.get_nearest_cluster(query_vector)
                
                # 해당 클러스터의 문서 ID 가져오기
                cluster_doc_ids = self.clustering.get_cluster_ids(nearest_cluster)
                
                # 클러스터 내 문서가 없거나 너무 적으면 일반 검색으로 대체
                if len(cluster_doc_ids) < top_k:
                    logger.warning(f"Cluster {nearest_cluster} has only {len(cluster_doc_ids)} documents, falling back to regular search")
                    use_clustering = False
                else:
                    # 그 클러스터 내에서만 검색
                    logger.info(f"Searching in cluster {nearest_cluster} with {len(cluster_doc_ids)} documents (similarity: {cluster_score:.4f})")
                    results = self.collection.query(
                        query_embeddings=[query_vector],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"],
                        where={"cluster_id": nearest_cluster}
                    )
            
            # 일반 검색 (클러스터링 비활성화 또는 대체 시)
            if not use_clustering or not self.clustering.cluster_centers:
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            # 결과 포맷팅
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # 거리를 유사도 점수로 변환 (ChromaDB는 거리 반환)
                    # 코사인의 경우 거리를 유사도로 변환 (1 - 거리)
                    if self.metric_type == "cosine":
                        score = 1 - results["distances"][0][i]
                    else:
                        # 다른 메트릭의 경우 거리의 음수 사용
                        score = -results["distances"][0][i]
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "question": results["documents"][0][i],
                        "answer": results["metadatas"][0][i]["answer"],
                        "score": score,
                        "cluster_id": results["metadatas"][0][i].get("cluster_id")
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # 클러스터링 검색 실패 시 일반 검색 시도
            if use_clustering:
                logger.info("Falling back to non-clustered search")
                return self.retrieve(query=query, query_vector=query_vector, top_k=top_k, use_clustering=False)
            raise
    
    def get_document_count(self) -> int:
        """
        컬렉션의 총 문서 수 가져오기
        
        Returns:
            문서 수
        """
        try:
            # 컬렉션 수 가져오기
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_cluster_stats(self) -> Dict:
        """
        클러스터링 통계 가져오기
        
        Returns:
            클러스터링 통계 사전
        """
        if not self.clustering.cluster_to_ids:
            return {"status": "Clustering not available"}
            
        cluster_sizes = [len(ids) for ids in self.clustering.cluster_to_ids.values()]
        return {
            "num_clusters": len(self.clustering.cluster_centers),
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
            "total_documents": sum(cluster_sizes)
        }
            
    def batch_process_questions(self, questions: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        캐싱을 활용한 질문 배치 처리로 임베딩 가져오기
        
        Args:
            questions: 처리할 질문 리스트
            batch_size: 처리할 배치 크기
            
        Returns:
            임베딩 벡터 리스트
        """
        if not self.embedder:
            raise ValueError("No embedder provided for generating embeddings")
        
        # 임베딩이 필요한 질문 식별
        questions_to_embed = []
        question_hashes = []
        result_map = {}
        
        for i, question in enumerate(questions):
            question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            
            if question_hash in self.embedding_cache:
                # 캐시된 임베딩 사용
                result_map[i] = self.embedding_cache[question_hash]
            else:
                # 임베딩 생성 필요
                questions_to_embed.append(question)
                question_hashes.append(question_hash)
                result_map[i] = None
        
        # 임베딩이 필요한 질문을 배치로 처리
        if questions_to_embed:
            logger.info(f"Generating embeddings for {len(questions_to_embed)} questions")
            
            for i in range(0, len(questions_to_embed), batch_size):
                batch = questions_to_embed[i:i+batch_size]
                batch_hashes = question_hashes[i:i+batch_size]
                
                # 배치에 대한 임베딩 생성
                batch_embeddings = self.embedder.get_embeddings_batch(batch)
                
                # 캐시 업데이트
                for j, embedding in enumerate(batch_embeddings):
                    hash_key = batch_hashes[j]
                    self.embedding_cache[hash_key] = embedding
                
                logger.info(f"Processed batch of {len(batch)} questions")
            
            # 업데이트된 캐시 저장
            with open(self.cache_file, "w") as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Updated embedding cache with {len(self.embedding_cache)} entries")
            
            # 결과_맵에서 누락된 임베딩 채우기
            for i, question in enumerate(questions):
                if result_map[i] is None:
                    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
                    result_map[i] = self.embedding_cache[question_hash]
        
        # 원래 순서로 결과 구성
        results = [result_map[i] for i in range(len(questions))]
        return results
