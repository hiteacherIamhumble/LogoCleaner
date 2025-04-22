<template>
  <div class="example-gallery">
    <div class="gallery-header">
      <h2 class="gallery-title">{{ title }}</h2>
      <p class="gallery-description">{{ description }}</p>
    </div>
    
    <div class="gallery-grid">
      <div 
        v-for="(example, index) in examples" 
        :key="index"
        class="example-card"
        @click="selectExample(example)"
      >
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
</template>

<script>
export default {
  name: 'ExampleGallery',
  props: {
    title: {
      type: String,
      default: 'Example Gallery'
    },
    description: {
      type: String,
      default: 'See examples of logo removal in action'
    },
    examples: {
      type: Array,
      required: true
    }
  },
  methods: {
    selectExample(example) {
      this.$emit('select', example);
    }
  }
}
</script>

<style scoped>
.example-gallery {
  margin-bottom: 4rem;
}

.gallery-header {
  text-align: center;
  margin-bottom: 3rem;
}

.gallery-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: #1F2937;
}

.gallery-description {
  font-size: 1.2rem;
  color: #6B7280;
  max-width: 700px;
  margin: 0 auto;
}

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 2rem;
}

.example-card {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
  background: white;
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

@media (max-width: 768px) {
  .gallery-title {
    font-size: 2rem;
  }
  
  .gallery-description {
    font-size: 1rem;
  }
  
  .example-images {
    padding: 1rem;
  }
  
  .example-caption {
    padding: 1rem;
  }
}
</style>
