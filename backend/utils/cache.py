import time

class PredictionCache:
    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self._cache[key]
        return None

    def set(self, key, data):
        self._cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
    def clear(self):
        self._cache.clear()

global_cache = PredictionCache()
