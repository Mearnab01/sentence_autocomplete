import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('dl_mode'); // 'dl_mode' or 'llm_mode'
  const [suggestions, setSuggestions] = useState([]);
  const [latency, setLatency] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const abortControllerRef = useRef(null);

  useEffect(() => {
    // Debounce the call by 300ms
    const delayDebounceFn = setTimeout(() => {
      if (query.trim().length > 0) {
        fetchSuggestions(query, mode);
      } else {
        setSuggestions([]);
        setLatency(0);
      }
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [query, mode]);

  const fetchSuggestions = async (text, currentMode) => {
    setIsLoading(true);
    setError('');
    
    // Abort previous request if still in flight
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          mode: currentMode,
          top_k: 3
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setSuggestions([]);
      } else {
        setSuggestions(data.suggestions || []);
        setLatency(data.latency_ms || 0);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Fetch error:', err);
        setError('Failed to fetch suggestions');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(query + (query.endsWith(' ') ? '' : ' ') + suggestion + ' ');
    setSuggestions([]);
  };

  const toggleMode = () => {
    setMode(prev => prev === 'dl_mode' ? 'llm_mode' : 'dl_mode');
    setSuggestions([]);
    setLatency(0);
  };

  return (
    <div className="app-container">
      <h1>Smart Autocomplete</h1>
      
      <div className="controls">
        <div className="toggle-group">
          <label>Deep Learning (LSTM)</label>
          <label className="switch">
            <input 
              type="checkbox" 
              checked={mode === 'llm_mode'} 
              onChange={toggleMode} 
            />
            <span className="slider"></span>
          </label>
          <label>LLM API (HuggingFace)</label>
        </div>
      </div>

      <div className="search-container">
        <input
          type="text"
          className="search-input"
          placeholder="Start typing deeply intelligent sentences..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        
        {isLoading && (
          <div style={{ position: 'absolute', right: '20px', top: '20px' }}>
            <div className="loader"></div>
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="suggestions-dropdown">
            {suggestions.map((suggestion, idx) => (
              <div 
                key={idx} 
                className="suggestion-item"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                <span className="suggestion-icon">🪄</span>
                <span className="suggestion-text">... {suggestion}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div style={{ color: '#ef4444', marginTop: '1rem', background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '8px' }}>
          {error}
        </div>
      )}

      <div className="stats-container">
        <div className="stat-box">
          <span className="stat-label">Current Mode</span>
          <span className="stat-value highlight">
            {mode === 'dl_mode' ? 'PyTorch LSTM' : 'HF LLM'}
          </span>
        </div>
        <div className="stat-box">
          <span className="stat-label">Inference Latency</span>
          <span className="stat-value">{latency} ms</span>
        </div>
      </div>
    </div>
  );
}

export default App;
