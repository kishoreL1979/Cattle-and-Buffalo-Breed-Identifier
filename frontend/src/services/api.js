import axios from 'axios';

let rawUrl = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '');

// Automatic resolution for cloud deployment
if (!rawUrl && typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
  const host = window.location.hostname;
  if (host.includes('onrender.com')) {
    // Render multi-service mode
    const backendHost = host.replace(/-frontend(-[a-z0-9]+)?/, '$1').replace('.onrender.com', '');
    rawUrl = `https://${backendHost.replace('-frontend', '')}-backend.onrender.com/api`;
  } else {
    // Vercel cloud frontend: connect to active backend service
    rawUrl = 'https://cattle-and-buffalo-breed-identifier.onrender.com/api';
  }
}

if (!rawUrl) {
  rawUrl = 'http://127.0.0.1:8000/api';
}

if (!rawUrl.endsWith('/api')) {
  rawUrl = `${rawUrl}/api`;
}

const API_BASE_URL = rawUrl;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 90000,
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
      throw new Error('Backend server unavailable or waking up. If deployed on Render, please wait 30 seconds for cold-start and try again.');
    } else {
      throw new Error(error.message || 'An unexpected error occurred.');
    }
  }
};

export default api;
