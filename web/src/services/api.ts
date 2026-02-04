const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const fetchPrediction = async (symbol) => {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ symbol, timeframe: '15m' }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch prediction');
  }
  
  return response.json();
};

export const fetchKlines = async (symbol, timeframe = '15m', limit = 500) => {
  const response = await fetch(
    `${API_BASE_URL}/api/klines?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&limit=${limit}`
  );
  
  if (!response.ok) {
    throw new Error('Failed to fetch klines');
  }
  
  return response.json();
};

export const fetchPerformance = async (symbol) => {
  const url = symbol 
    ? `${API_BASE_URL}/api/performance?symbol=${encodeURIComponent(symbol)}`
    : `${API_BASE_URL}/api/performance`;
    
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error('Failed to fetch performance');
  }
  
  return response.json();
};
