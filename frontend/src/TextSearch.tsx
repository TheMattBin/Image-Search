// src/components/TextSearch.tsx
import React, { useState } from "react";
import axios from "axios";

export const TextSearch: React.FC = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = async () => {
    const res = await axios.get("http://localhost:8000/search-by-text/", {
      params: { query }
    });
    setResults(res.data);
  };

  return (
    <div>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      <button onClick={handleSearch}>Search</button>
      <ul>
        {results.map((item, idx) => (
          <li key={idx}>{item.caption || item.description}</li>
        ))}
      </ul>
    </div>
  );
};