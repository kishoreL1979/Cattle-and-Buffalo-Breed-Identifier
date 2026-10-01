import axios from 'axios';

let rawUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api').trim().replace(/\/+$/, '');
if (!rawUrl.endsWith('/api')) {
  rawUrl = `${rawUrl}/api`;
}
const API_BASE_URL = rawUrl;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const predictBreed = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server returned error status
      const message = error.response.data?.detail || 'Prediction failed. Please try again.';
      throw new Error(message);
    } else if (error.request) {
      // Request made but no response received
      throw new Error('Backend server unavailable. Please ensure FastAPI server is running at http://127.0.0.1:8000.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred.');
    }
  }
};

export default api;
