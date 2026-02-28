import { useState } from 'react';
import './App.css';
import { searchByText, uploadImage, searchByImage } from './api';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [caption, setCaption] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setMessage(null);
    try {
      const res = await uploadImage(file);
      setCaption(res.caption);
      setMessage('Image indexed successfully!');
      setResults([]);
    } catch (e) {
      setMessage('Failed to index image.');
    }
    setLoading(false);
  };

  const handleTextSearch = async () => {
    setLoading(true);
    setMessage(null);
    setCaption(null);
    try {
      const res = await searchByText(query);
      setResults(res);
      setMessage(res.length ? null : 'No results found.');
    } catch (e) {
      setMessage('Text search failed.');
    }
    setLoading(false);
  };

  const handleImageSearch = async () => {
    if (!file) return;
    setLoading(true);
    setMessage(null);
    setCaption(null);
    try {
      const res = await searchByImage(file);
      setResults(res);
      setMessage(res.length ? null : 'No results found.');
    } catch (e) {
      setMessage('Image search failed.');
    }
    setLoading(false);
  };

  return (
    <div className="modern-app-container">
      <h1>Visual Search Engine</h1>
      <div className="modern-main-row">
        <div className="modern-main-col modern-main-col-left">
          <div className="modern-section">
            <h2>Index Image</h2>
            <input type="file" onChange={e => setFile(e.target.files ? e.target.files[0] : null)} />
            <button className="modern-btn" onClick={handleUpload} disabled={!file || loading}>Index Image</button>
          </div>
          <div className="modern-section">
            <h2>Search</h2>
            <div className="modern-search-row">
              <input
                className="modern-input"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search by text"
                disabled={loading}
              />
              <button className="modern-btn" onClick={handleTextSearch} disabled={!query || loading}>Search by Text</button>
            </div>
            <div className="modern-search-row">
              <input type="file" onChange={e => setFile(e.target.files ? e.target.files[0] : null)} />
              <button className="modern-btn" onClick={handleImageSearch} disabled={!file || loading}>Search by Image</button>
            </div>
          </div>
        </div>
        <div className="modern-main-col modern-main-col-right">
          {loading && <div className="modern-loading">Loading...</div>}
          {message && <div className="modern-message">{message}</div>}
          {caption && <div className="modern-caption">Caption: {caption}</div>}
          <div className="modern-results">
            {results.map((item, idx) => (
              <div className="modern-result-card" key={idx}>
                <div className="modern-result-caption">{item.caption || item.description || 'No caption'}</div>
                {item.image_url && <img src={item.image_url} alt="result" className="modern-result-img" />}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
