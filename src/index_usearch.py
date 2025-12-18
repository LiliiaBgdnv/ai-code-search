import numpy as np

try:
    from usearch.index import Index as _USearchIndex

    _BACKEND = "usearch"
except Exception:
    # If USearch is not installed or fails to import, fall back to sklearn
    _USearchIndex = None
    _BACKEND = "sklearn"

if _BACKEND == "sklearn":
    from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    """
    Uniform wrapper around a vector index.

    - If USearch is available, uses it for fast ANN search.
    - Else falls back to sklearn brute-force cosine over a matrix.
    """

    def __init__(
        self,
        ndim: int,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        self.ndim = ndim
        self.metric = metric

        if _BACKEND == "usearch":
            # USearch index, HNSW-like parameters
            self._idx = _USearchIndex(
                ndim=ndim,
                metric="cos" if metric in ("cos", "cosine") else metric,
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            )
            self._backend = "usearch"
        else:
            # Simple brute-force fallback
            self._ids = None  # shape: (n,)
            self._vecs = None  # shape: (n, d)
            self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self._backend = "sklearn"

    def add(self, ids: np.ndarray, vectors: np.ndarray):
        """Add vectors to the index.

        ids: array-like of int (unique labels for each vector)
        vectors: (n, d) float32, contiguous
        """
        ids64 = np.asarray(ids, dtype=np.int64).ravel()
        vecs = np.asarray(vectors, dtype=np.float32)

        # Accept 1D and reshape if needed
        if vecs.ndim == 1:
            if vecs.size % self.ndim != 0:
                raise ValueError(
                    f"Vector size {vecs.size} not divisible by ndim={self.ndim}"
                )
            vecs = vecs.reshape(-1, self.ndim)
        elif vecs.ndim == 2 and vecs.shape[1] != self.ndim:
            raise ValueError(
                f"vectors.shape[1]={vecs.shape[1]} != index ndim={self.ndim}"
            )
        vecs32 = np.ascontiguousarray(vecs, dtype=np.float32)

        # Basic validation
        if not np.isfinite(vecs32).all():
            raise ValueError("Vectors contain NaN/Inf values")

        # If lengths do not match, fall back to 0..n-1 ids (safe default)
        if len(ids64) != vecs32.shape[0]:
            ids64 = np.arange(vecs32.shape[0], dtype=np.int64)

        if self._backend == "usearch":
            try:
                # Newer USearch (>=2.10)
                self._idx.add(keys=ids64, vectors=vecs32)
            except TypeError:
                # Older USearch (<2.10)
                self._idx.add(labels=ids64, vectors=vecs32)
        else:
            # Store full matrix for brute-force cosine
            self._ids = ids64
            self._vecs = vecs32
            self._nn.fit(self._vecs)

    def search(self, queries: np.ndarray, k: int = 10):
        """Search top-k neighbors.

        Returns:
            labels: (n_queries, k) int64
            dists : (n_queries, k) float32 (cosine distance if sklearn)
        """
        if self._backend == "usearch":
            vecs32 = queries.astype(np.float32, copy=False)
            try:
                res = self._idx.search(vectors=vecs32, count=k)  # >= 2.10
            except TypeError:
                res = self._idx.search(vectors=vecs32, k=k)  # older

            # USearch can return a tuple or an object; normalize to arrays
            if isinstance(res, (list, tuple)):
                keys, dists = res[0], res[1]
            else:
                keys = getattr(res, "keys", None)
                dists = getattr(res, "distances", None)
                if keys is None or dists is None:
                    raise TypeError("Unexpected return type from USearch Index.search")

            keys = np.asarray(keys)
            dists = np.asarray(dists)

            if keys.ndim == 1:
                keys = keys[None, :]
            if dists.ndim == 1:
                dists = dists[None, :]

            return keys, dists

        else:
            # sklearn brute-force cosine
            dist, idx = self._nn.kneighbors(
                queries.astype(np.float32, copy=False),
                n_neighbors=k,
                return_distance=True,
            )
            labels = np.where(idx >= 0, self._ids[idx], -1)
            return labels, dist

    def save(self, path: str):
        """Save index to disk (only for USearch backend)."""
        if self._backend == "usearch":
            self._idx.save(path)

    def load(self, path: str):
        """Load index from disk (only for USearch backend)."""
        if self._backend == "usearch":
            self._idx.load(path)
        return self
