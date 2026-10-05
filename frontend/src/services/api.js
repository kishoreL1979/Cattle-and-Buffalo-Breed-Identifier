import axios from 'axios';

// Candidate backend API URLs for automatic failover across Render and Local environments
const CANDIDATE_URLS = [
  import.meta.env.VITE_API_URL,
  'https://breedvision-backend.onrender.com/api',
  'https://cattle-and-buffalo-breed-identifier.onrender.com/api',
  'http://127.0.0.1:8000/api'
].filter(Boolean).map(url => {
  let clean = url.trim().replace(/\/+$/, '');
  return clean.endsWith('/api') ? clean : `${clean}/api`;
});

// Remove duplicates while maintaining order
const UNIQUE_URLS = [...new Set(CANDIDATE_URLS)];

let activeBaseUrl = UNIQUE_URLS[0];

export const predictBreed = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  let lastError = null;

  for (const baseUrl of UNIQUE_URLS) {
    try {
      const client = axios.create({
        baseURL: baseUrl,
        timeout: 90000,
      });

      const response = await client.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Save working URL for subsequent calls
      activeBaseUrl = baseUrl;
      return response.data;
    } catch (error) {
      lastError = error;
      if (error.response) {
        // Server responded with an explicit status error (400, 422, 500, etc.)
        const message = error.response.data?.detail || 'Prediction failed. Please try again.';
        throw new Error(message);
      }
      // If network connection error, continue to try next candidate backend URL
    }
  }

  // If all backend candidate URLs failed to connect
  throw new Error(
    'Backend server unavailable or waking up. If deployed on Render, please wait 30 seconds for cold-start and click Identify Breed again.'
  );
};

export default axios.create({ baseURL: activeBaseUrl, timeout: 90000 });
