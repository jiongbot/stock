import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchPrediction = async (symbol) => {
  const response = await api.get(`/api/predict/${symbol.replace('/', '-')}`);
  return response.data;
};

export const fetchHistory = async (symbol, limit = 500) => {
  const response = await api.get(`/api/history/${symbol.replace('/', '-')}?limit=${limit}`);
  return response.data;
};

export const fetchPerformance = async () => {
  const response = await api.get('/api/performance');
  return response.data;
};

export const fetchIndicators = async (symbol) => {
  const response = await api.get(`/api/indicators/${symbol.replace('/', '-')}`);
  return response.data;
};

export const updateData = async (symbol) => {
  const response = await api.post(`/api/update/${symbol.replace('/', '-')}`);
  return response.data;
};

export default api;
