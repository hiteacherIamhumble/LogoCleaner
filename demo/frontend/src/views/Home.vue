<template>
  <div class="home">
    <!-- Hero Section -->
    <section class="hero">
      <div class="container">
        <div class="hero-content">
          <h1 class="hero-title">
            Remove Logos from Images
            <span class="hero-highlight">in Seconds</span>
          </h1>
          <p class="hero-subtitle">
            Our AI-powered tool detects and cleanly removes logos from any image,
            giving you pristine visuals without watermarks or branding.
          </p>
          
          <!-- Upload Container -->
          <div class="upload-container" :class="{ 'uploading': isUploading }">
            <UploadDropzone
              v-model="selectedFile"
              @error="handleUploadError"
              title="Upload an Image"
              description="Drag & drop or click to browse"
              formats="Supports: JPG, PNG, GIF"
            />
            
            <button 
              class="btn btn-primary btn-lg btn-block process-btn" 
              :disabled="!selectedFile || isUploading"
              @click="processImage"
            >
              <span v-if="!isUploading">Remove Logo</span>
              <span v-else class="spinner"></span>
            </button>
          </div>
        </div>
      </div>
      
      <!-- Hero Shapes -->
      <div class="hero-shape hero-shape-1"></div>
      <div class="hero-shape hero-shape-2"></div>
      <div class="hero-shape hero-shape-3"></div>
    </section>
    
    <!-- Features Section -->
    <section id="features" class="features">
      <div class="container">
        <h2 class="section-title">Why Choose LogoCleaner?</h2>
        
        <div class="features-grid">
          <div class="feature-card" v-for="(feature, index) in features" :key="index">
            <div class="feature-icon" :style="{ background: feature.iconBg }">
              <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path :d="feature.icon"></path>
              </svg>
            </div>
            <h3 class="feature-title">{{ feature.title }}</h3>
            <p class="feature-desc">{{ feature.description }}</p>
          </div>
        </div>
      </div>
    </section>
    
    <!-- Examples Section -->
    <section id="examples" class="examples">
      <div class="container">
        <h2 class="section-title">See It In Action</h2>
        
        <div class="examples-grid">
          <div class="example-card" v-for="(example, index) in examples" :key="index">
            <div class="example-images">
              <div class="example-image-container">
                <img :src="example.before" alt="Before" class="example-image">
                <span class="example-label">Before</span>
              </div>
              
              <div class="example-arrow">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </div>
              
              <div class="example-image-container">
                <img :src="example.after" alt="After" class="example-image">
                <span class="example-label">After</span>
              </div>
            </div>
            <p class="example-caption">{{ example.caption }}</p>
          </div>
        </div>
      </div>
    </section>
    
    <!-- CTA Section -->
    <section class="cta">
      <div class="container">
        <div class="cta-content">
          <h2 class="cta-title">Ready to Remove Unwanted Logos?</h2>
          <p class="cta-subtitle">Try our advanced AI tool now - it only takes seconds</p>
          <a href="#" class="btn btn-primary btn-lg" @click.prevent="scrollToTop">Upload an Image Now</a>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useToast } from 'vue-toastification';
import UploadDropzone from '@/components/UploadDropzone.vue';
import api from '@/api';

// Component setup
const router = useRouter();
const toast = useToast();

// Reactive data
const isUploading = ref(false);
const selectedFile = ref(null);

// Features array
const features = [
  {
    title: 'AI-Powered Detection',
    description: 'Our advanced machine learning algorithms accurately identify logos even in complex backgrounds.',
    icon: 'M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18',
    iconBg: 'rgba(99, 102, 241, 0.1)'
  },
  {
    title: 'Seamless Removal',
    description: 'We don\'t just erase logos - we intelligently fill the space with matching background content.',
    icon: 'M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6 M15 3h6v6 M10 14L21 3',
    iconBg: 'rgba(139, 92, 246, 0.1)'
  },
  {
    title: 'High Resolution Support',
    description: 'Process images of any size with high-quality results that maintain original resolution.',
    icon: 'M15 3h6v6 M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6 M21 14v5a2 2 0 0 1-2 2h-5 M3 9l9 9 M3 19l6-6 M14 10l7-7',
    iconBg: 'rgba(234, 88, 12, 0.1)'
  }
];

// Example images
const examples = [
  {
    before: new URL('@/assets/example1-before.jpg', import.meta.url).href,
    after: new URL('@/assets/example1-after.jpg', import.meta.url).href,
    caption: 'Trademark cleaned from car photo from different angle'
  },
  {
    before: new URL('@/assets/example2-before.jpg', import.meta.url).href,
    after: new URL('@/assets/example2-after.jpg', import.meta.url).href,
    caption: 'Ford Trademark cleaned from car photo'
  },
  {
    before: new URL('@/assets/example3-before.jpg', import.meta.url).href,
    after: new URL('@/assets/example3-after.jpg', import.meta.url).href,
    caption: 'PolyU logo cleaned from building photo'
  }
];

// Methods
function handleUploadError(message) {
  toast.error(message);
}

async function processImage() {
  if (!selectedFile.value) {
    toast.error('Please select an image first');
    return;
  }
  
  isUploading.value = true;
  
  try {
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile.value);
    
    console.log('Processing image:', selectedFile.value.name, 'Size:', selectedFile.value.size);
    
    // Upload the file
    const uploadResponse = await api.uploadImage(formData);
    
    console.log('Upload response:', uploadResponse.data);
    
    if (uploadResponse.data.success) {
      // Create a job ID for the new process
      const filepath = uploadResponse.data.filepath;
      console.log('Creating job with filepath:', filepath);
      
      try {
        const jobResponse = await api.createJob({
          filepath: filepath
        });
        
        console.log('Job response:', jobResponse.data);
        
        // Navigate to results page immediately after getting job ID
        router.push(`/results/${jobResponse.data.job_id}`);
      } catch (jobError) {
        console.error('Error creating job:', jobError);
        if (jobError.response && jobError.response.data) {
          toast.error(`Error: ${jobError.response.data.error || 'Failed to create processing job'}`);
        } else {
          toast.error('Failed to create processing job. Please try again.');
        }
      }
    } else {
      toast.error(uploadResponse.data.error || 'Upload failed');
    }
  } catch (error) {
    console.error('Error processing image:', error);
    if (error.response && error.response.data) {
      toast.error(`Error: ${error.response.data.error || 'Failed to process image'}`);
    } else {
      toast.error('Failed to process image. Please try again.');
    }
  } finally {
    isUploading.value = false;
  }
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}
</script>

<style scoped>
.home {
  overflow-x: hidden;
}

/* Hero Section */
.hero {
  position: relative;
  padding: 6rem 0 8rem;
  background: white;
  overflow: hidden;
}

.hero-content {
  text-align: center;
  max-width: 800px;
  margin: 0 auto;
  position: relative;
  z-index: 2;
}

.hero-title {
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  line-height: 1.2;
  color: #1F2937;
}

.hero-highlight {
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  font-size: 1.2rem;
  color: #6B7280;
  line-height: 1.6;
  margin-bottom: 3rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.hero-shape {
  position: absolute;
  border-radius: 50%;
}

.hero-shape-1 {
  width: 600px;
  height: 600px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
  top: -300px;
  right: -200px;
}

.hero-shape-2 {
  width: 400px;
  height: 400px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.03), rgba(139, 92, 246, 0.03));
  bottom: -200px;
  left: -200px;
}

.hero-shape-3 {
  width: 200px;
  height: 200px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.07), rgba(139, 92, 246, 0.07));
  top: 300px;
  right: 300px;
}

/* Upload Container */
.upload-container {
  max-width: 600px;
  margin: 0 auto;
}

.process-btn {
  height: 56px;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.process-btn:disabled {
  background: #D1D5DB;
  cursor: not-allowed;
  box-shadow: none;
}

/* Features Section */
.features {
  padding: 6rem 0;
  background: #F9FAFB;
}

.section-title {
  text-align: center;
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 4rem;
  color: #1F2937;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
}

.feature-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  color: #6366F1;
}

.feature-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #1F2937;
}

.feature-desc {
  color: #6B7280;
  line-height: 1.6;
}

/* Examples Section */
.examples {
  padding: 6rem 0;
  background: white;
}

.examples-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 3rem;
}

.example-card {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
}

.example-images {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem;
  background: #F9FAFB;
}

.example-image-container {
  position: relative;
  width: 45%;
}

.example-image {
  width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
}

.example-label {
  position: absolute;
  bottom: -10px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  color: #6B7280;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.example-arrow {
  color: #6366F1;
}

.example-caption {
  padding: 1.5rem;
  text-align: center;
  font-weight: 500;
  color: #4B5563;
}

/* CTA Section */
.cta {
  padding: 6rem 0;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  color: white;
}

.cta-content {
  text-align: center;
  max-width: 700px;
  margin: 0 auto;
}

.cta-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
}

.cta-subtitle {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  opacity: 0.9;
}

.cta .btn-primary {
  background: white;
  color: #6366F1;
  box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
}

.cta .btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(255, 255, 255, 0.4);
}

/* Responsive styles */
@media (max-width: 768px) {
  .hero {
    padding: 4rem 0 6rem;
  }
  
  .hero-title {
    font-size: 2.2rem;
  }
  
  .hero-subtitle {
    font-size: 1rem;
  }
  
  .section-title {
    font-size: 2rem;
    margin-bottom: 3rem;
  }
  
  .features-grid, 
  .examples-grid {
    grid-template-columns: 1fr;
  }
  
  .cta-title {
    font-size: 2rem;
  }
  
  .cta-subtitle {
    font-size: 1rem;
  }
}
</style>