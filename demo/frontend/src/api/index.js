import axios from 'axios';

// Create axios instance for API calls
const apiClient = axios.create({
  // Base URL will be automatically proxied through Vite in development
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  timeout: 120000 // 2 minutes timeout for longer operations
});

// Create a separate client with longer timeout for image generation
const longOperationClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  timeout: 300000 // 5 minutes timeout for OpenAI image generation
});

export default {
  // Image Upload
  uploadImage(formData) {
    console.log('Uploading image:', formData.get('file').name);
    return apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }).catch(error => {
      console.error('Upload error:', error.response ? error.response.data : error.message);
      throw error;
    });
  },
  
  // Create a new job (just creates directory structure and returns job ID)
  createJob(data) {
    return apiClient.post('/create-job', data);
  },
  
  // Start Processing Image (only detect the logo, don't generate the result)
  startProcessing(data) {
    return apiClient.post('/start-process', data);
  },
  
  // Continue Processing Image (generate the final result)
  generateImage(jobId) {
    // Add more detailed error handling and use the longer timeout client
    return longOperationClient.post(`/generate/${jobId}`, {})
      .catch(error => {
        console.error('API Error details:', error.response ? error.response.data : error.message);
        throw error;
      });
  },
  
  
  // Get Results
  getResults(jobId) {
    return apiClient.get(`/results/${jobId}`);
  },
  
  // Get Examples
  getExamples() {
    return apiClient.get('/examples');
  },
  
  // Get URLs for a specific job
  getResultUrls(jobId, timestamp = Date.now()) {
    return {
      original: `/api/results/${jobId}/original.png`,
      bbox: `/api/results/${jobId}/original_with_bbox.png?t=${timestamp}`,  // Add cache-busting query param
      mask: `/api/results/${jobId}/refined_mask_overlay.png?t=${timestamp}`,
      removed: `/api/results/${jobId}/logo_removed.png?t=${timestamp}`
    };
  }
};
