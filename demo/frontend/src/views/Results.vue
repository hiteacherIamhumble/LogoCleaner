<template>
  <div class="results-page">
    <!-- Header Section -->
    <section class="results-header">
      <div class="container">
        <h1 class="results-title">Logo Removal Process</h1>
        <p class="results-subtitle">
          Watch our AI detect and remove logos from your image
        </p>
      </div>
    </section>
    
    <!-- Status Section -->
    <section class="status-section">
      <div class="container">
        <ProcessingStatus :steps="steps" :currentStep="currentStep" />
      </div>
    </section>
    
    <!-- Results Display Section -->
    <section class="results-display">
      <div class="container">
        <div class="results-grid">
          <!-- Original Image -->
          <div class="result-card" style="grid-area: original;">
            <h3 class="result-title">Original Image</h3>
            <div class="image-container">
              <div v-if="isLoading" class="loading-placeholder">
                <div class="spinner large"></div>
              </div>
              <img 
                v-else 
                :src="results.original" 
                alt="Original Image" 
                class="result-image"
                @error="handleImageError($event, 'original')"
              >
            </div>
          </div>
          
          <!-- Logo Detection with Bounding Box -->
          <div class="result-card" style="grid-area: detection;">
            <h3 class="result-title">Logo Detection</h3>
            <div class="image-container">
              <div v-if="!results.bbox || isLoading || currentStep < 1" class="loading-placeholder">
                <div class="spinner large" v-if="isLoading || currentStep < 1"></div>
                <div class="placeholder-message" v-else-if="currentStep < 1">Waiting to detect logo...</div>
                <div class="error-message" v-else-if="hasErrors.bbox">
                  Failed to load image
                </div>
              </div>
              <img 
                v-else 
                :src="results.bbox" 
                alt="Logo Detection" 
                class="result-image"
                @error="handleImageError($event, 'bbox')"
              >
            </div>
            <div class="result-caption">
              <div class="caption-icon" :class="{ 'inactive': currentStep < 1 }">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
              </div>
              <p :class="{ 'inactive-text': currentStep < 1 }">AI identified logo location using advanced object detection</p>
            </div>
          </div>
          
          <!-- Logo Mask -->
          <div class="result-card" style="grid-area: mask;">
            <h3 class="result-title">Logo Mask</h3>
            <div class="image-container">
              <div v-if="!results.mask || isLoading || currentStep < 2" class="loading-placeholder">
                <div class="spinner large" v-if="isLoading || (currentStep >= 1 && currentStep < 2)"></div>
                <div class="placeholder-message" v-else-if="currentStep < 2">Waiting to create mask...</div>
                <div class="error-message" v-else-if="hasErrors.mask">
                  Failed to load image
                </div>
              </div>
              <img 
                v-else 
                :src="results.mask" 
                alt="Logo Mask" 
                class="result-image"
                @error="handleImageError($event, 'mask')"
              >
            </div>
            <div class="result-caption">
              <div class="caption-icon" :class="{ 'inactive': currentStep < 2 }">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                  <path d="M2 17l10 5 10-5"></path>
                  <path d="M2 12l10 5 10-5"></path>
                </svg>
              </div>
              <p :class="{ 'inactive-text': currentStep < 2 }">Precise segmentation mask created to isolate the logo area</p>
            </div>
          </div>
          
          <!-- Removed Logo Image -->
          <div class="result-card result-card-highlight" style="grid-area: result;">
            <h3 class="result-title">Final Result</h3>
            <div class="image-container">
              <div v-if="!results.removed || isLoading || currentStep < 3" class="loading-placeholder">
                <div class="spinner large" v-if="isLoading || (currentStep >= 2 && currentStep < 3)"></div>
                <div class="placeholder-message" v-else-if="currentStep < 3">Waiting to remove logo...</div>
                <div class="error-message" v-else-if="hasErrors.removed">
                  Failed to load image
                </div>
              </div>
              <img 
                v-else 
                :src="results.removed" 
                alt="Logo Removed" 
                class="result-image"
                @error="handleImageError($event, 'removed')"
              >
            </div>
            <div class="result-caption">
              <div class="caption-icon" :class="{ 'inactive': currentStep < 3 }">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                </svg>
              </div>
              <p :class="{ 'inactive-text': currentStep < 3 }">Logo successfully removed and background seamlessly filled</p>
            </div>
            <button 
              v-if="currentStep >= 3" 
              class="btn btn-primary btn-block download-btn"
              @click="downloadResult"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Download Image
            </button>
          </div>
        </div>
        
        <div class="action-buttons">
          <button class="btn btn-secondary" @click="goBack">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="19" y1="12" x2="5" y2="12"></line>
              <polyline points="12 19 5 12 12 5"></polyline>
            </svg>
            Process Another Image
          </button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useToast } from 'vue-toastification';
import gsap from 'gsap';
import ProcessingStatus from '@/components/ProcessingStatus.vue';
import api from '@/api';

// Component setup
const router = useRouter();
const route = useRoute();
const toast = useToast();

// Props
const props = defineProps({
  jobId: {
    type: String,
    required: false
  }
});

// Reactive data
const isLoading = ref(true);
const currentStep = ref(0);
const steps = ref([
  {
    title: 'Detect Logo',
    description: 'AI analyzes the image to locate logos'
  },
  {
    title: 'Create Mask',
    description: 'Generate precise mask around the logo'
  },
  {
    title: 'Remove Logo',
    description: 'Erase the logo from the image'
  },
  {
    title: 'Refill Background',
    description: 'Intelligently fill the empty space'
  }
]);
const results = ref({
  original: null,
  bbox: null,
  mask: null,
  removed: null
});
const hasErrors = ref({
  original: false,
  bbox: false,
  mask: false,
  removed: false
});

// Lifecycle hooks
onMounted(() => {
  const id = props.jobId || route.params.jobId;
  
  if (!id) {
    router.push('/');
    return;
  }
  
  // First load the images that are already available
  results.value = api.getResultUrls(id);
  isLoading.value = false;
  
  // At this point, only show the original image by keeping currentStep at 0
  // (skip immediate animation start)
  currentStep.value = 0;
  
  // Set up polling to check for updates to detection images
  pollForUpdates(id);
  
  // Start the animation to set step to 0
  animateProgress();
  
  // After a short delay, initiate the generation process
  setTimeout(() => {
    startGeneration(id);
      }, 2000);
});

// Methods
async function pollForUpdates(jobId) {
  // Check every 1 second for updates to the detection images
  const pollInterval = setInterval(() => {
    checkImagesLoaded();
  }, 1000);
  
  // Clear the interval after 2 minutes (or when component unmounts)
  setTimeout(() => {
    clearInterval(pollInterval);
  }, 120000);
  
  // Ensure interval is cleared when component unmounts
  onUnmounted(() => {
    clearInterval(pollInterval);
  });
}

async function startGeneration(jobId) {
  try {
    console.log('Starting logo removal process...');
    await api.generateImage(jobId);
    console.log('Logo removal process initiated successfully');
    
    // Start checking for the final image periodically
    const checkFinalInterval = setInterval(() => {
      // Create a new image element to check if the removal image is ready
      const img = new Image();
      img.onload = () => {
        console.log('Final image is available');
        if (currentStep.value < 3) {
          gsap.to(currentStep, {
            value: 3,
            duration: 0.5,
            ease: 'power2.out'
          });
        }
        clearInterval(checkFinalInterval);
      };
      
      // Use cache-busting to bypass browser cache
      img.src = `${results.value.removed}?t=${Date.now()}`;
    }, 5000);
    
    // Clear the interval after 5 minutes to prevent indefinite checking
    setTimeout(() => {
      clearInterval(checkFinalInterval);
    }, 5 * 60 * 1000);
  } catch (error) {
    console.error('Error starting generation:', error);
    toast.warning('Logo detection is complete, but we encountered an issue with background removal.');
  }
}

function animateProgress() {
  // Keep step at 0 initially so the process can be shown progressing
  currentStep.value = 0;
  
  // Don't animate - just set the value directly to avoid triggering
  // any unwanted transitions at the start
}

function checkImagesLoaded() {
  // Add cache-busting to URLs
  const timestamp = Date.now();
  const bboxUrl = `${results.value.bbox.split('?')[0]}?t=${timestamp}`;
  const maskUrl = `${results.value.mask.split('?')[0]}?t=${timestamp}`;
  const removedUrl = `${results.value.removed.split('?')[0]}?t=${timestamp}`;
  
  // Reset any existing image references
  results.value = {
    ...results.value,
    bbox: bboxUrl,
    mask: maskUrl,
    removed: removedUrl
  };
  
  // Check if bbox image is loaded
  const bboxImg = new Image();
  bboxImg.onload = () => {
    // Only update if image actually exists (has non-zero dimensions and not transparent)
    if (bboxImg.width > 0 && bboxImg.height > 0) {
      // Check if the image has content (not a transparent placeholder)
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 1;
      tempCanvas.height = 1;
      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(bboxImg, 0, 0, 1, 1);
      const pixelData = ctx.getImageData(0, 0, 1, 1).data;
      
      // If the image has content (not fully transparent)
      if (pixelData[3] > 0) {
        // Image is loaded - update the step if needed
        if (currentStep.value < 1) {
          gsap.to(currentStep, {
            value: 1,
            duration: 0.5,
            ease: 'power2.out'
          });
        }
      }
    }
    
    // Now check if mask image is loaded
    const maskImg = new Image();
    maskImg.onload = () => {
      // Only update if image actually exists and has content
      if (maskImg.width > 0 && maskImg.height > 0) {
        // Check if the image has content (not a transparent placeholder)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 1;
        tempCanvas.height = 1;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(maskImg, 0, 0, 1, 1);
        const pixelData = ctx.getImageData(0, 0, 1, 1).data;
        
        // If the image has content (not fully transparent)
        if (pixelData[3] > 0) {
          // Mask image is loaded - update the step
          if (currentStep.value < 2) {
            gsap.to(currentStep, {
              value: 2,
              duration: 0.5,
              ease: 'power2.out'
            });
          }
        }
      }
    };
    maskImg.src = maskUrl;
  };
  bboxImg.src = bboxUrl;
  
  // Also check directly for removed image (in case it was already generated)
  const removedImg = new Image();
  removedImg.onload = () => {
    // Only update if image actually exists and has content
    if (removedImg.width > 0 && removedImg.height > 0) {
      // Check if the image has content (not a transparent placeholder)
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 1;
      tempCanvas.height = 1;
      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(removedImg, 0, 0, 1, 1);
      const pixelData = ctx.getImageData(0, 0, 1, 1).data;
      
      // If the image has content (not fully transparent)
      if (pixelData[3] > 0) {
        if (currentStep.value < 3) {
          gsap.to(currentStep, {
            value: 3,
            duration: 0.5,
            ease: 'power2.out'
          });
        }
      }
    }
  };
  removedImg.src = removedUrl;
}

function handleImageError(event, type) {
  event.target.style.display = 'none';
  hasErrors.value[type] = true;
  console.error(`Failed to load ${type} image`);
}

function downloadResult() {
  if (!results.value.removed) return;
  
  // Create a temporary link and trigger download
  const link = document.createElement('a');
  link.href = results.value.removed;
  link.download = 'logo-removed.png';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  toast.success('Download started');
}

function goBack() {
  router.push('/');
}
</script>

<style scoped>
.results-page {
  background: #F9FAFB;
  min-height: 100vh;
}

/* Header Section */
.results-header {
  background: white;
  padding: 3rem 0;
  text-align: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.results-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.results-subtitle {
  font-size: 1.2rem;
  color: #6B7280;
  max-width: 600px;
  margin: 0 auto;
}

/* Status Section */
.status-section {
  padding: 3rem 0;
}

/* Results Display Section */
.results-display {
  padding: 2rem 0 5rem;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: auto auto;
  grid-template-areas: 
    "original detection"
    "mask result";
  gap: 2rem;
  margin-bottom: 3rem;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

.result-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.result-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
}

.result-card-highlight {
  border: 2px solid rgba(99, 102, 241, 0.3);
}

.result-title {
  padding: 1.5rem;
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #1F2937;
  border-bottom: 1px solid #F3F4F6;
}

.image-container {
  width: 100%;
  height: 300px;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  transition: transform 0.3s ease;
}

.result-card:hover .result-image {
  transform: scale(1.02);
}

.loading-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #F9FAFB;
}

.error-message {
  color: #EF4444;
  font-size: 0.9rem;
}

.placeholder-message {
  color: #9CA3AF;
  font-size: 0.9rem;
  text-align: center;
}

.inactive {
  opacity: 0.4;
}

.inactive-text {
  color: #9CA3AF;
}

.result-caption {
  padding: 1.5rem;
  border-top: 1px solid #F3F4F6;
  display: flex;
  align-items: center;
}

.caption-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: rgba(99, 102, 241, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 1rem;
  color: #6366F1;
  flex-shrink: 0;
}

.download-btn {
  margin: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.spinner.large {
  width: 40px;
  height: 40px;
  border-width: 4px;
  border-color: rgba(99, 102, 241, 0.2);
  border-top-color: #6366F1;
}

.action-buttons {
  display: flex;
  justify-content: center;
}

.btn-secondary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Responsive styles */
@media (max-width: 768px) {
  .results-title {
    font-size: 2rem;
  }
  
  .results-subtitle {
    font-size: 1rem;
  }
  
  .results-grid {
    grid-template-columns: 1fr;
    grid-template-areas: 
      "original"
      "detection"
      "mask"
      "result";
    gap: 1.5rem;
  }
  
  .result-title {
    padding: 1rem;
    font-size: 1.1rem;
  }
  
  .image-container {
    height: 250px;
  }
  
  .result-caption {
    padding: 1rem;
  }
}

/* Medium sized screens */
@media (min-width: 769px) and (max-width: 1024px) {
  .results-grid {
    max-width: 900px;
  }
}
</style>