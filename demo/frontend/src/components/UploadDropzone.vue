<template>
  <div 
    class="dropzone"
    :class="{ 'active': isDragging, 'has-image': previewImage }"
    @dragover.prevent="onDragOver"
    @dragleave.prevent="onDragLeave"
    @drop.prevent="onDrop"
    @click="triggerFileInput"
  >
    <input 
      ref="fileInput" 
      type="file" 
      class="file-input" 
      accept="image/*"
      @change="onFileSelect"
    >
    
    <div v-if="!previewImage" class="upload-placeholder">
      <div class="upload-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
      </div>
      <h3 class="upload-title">{{ title || 'Upload an Image' }}</h3>
      <p class="upload-desc">{{ description || 'Drag & drop or click to browse' }}</p>
      <p class="upload-formats">{{ formats || 'Supports: JPG, PNG, GIF' }}</p>
    </div>
    
    <div v-else class="preview-container">
      <img :src="previewImage" alt="Preview" class="preview-image">
      <button class="remove-preview" @click.stop="removeImage">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';

const props = defineProps({
  title: String,
  description: String,
  formats: String,
  modelValue: File
});

const emit = defineEmits(['update:modelValue', 'error']);

const isDragging = ref(false);
const previewImage = ref(null);
const fileInput = ref(null);

function triggerFileInput() {
  fileInput.value.click();
}

function onDragOver(e) {
  isDragging.value = true;
}

function onDragLeave(e) {
  isDragging.value = false;
}

function onDrop(e) {
  isDragging.value = false;
  const files = e.dataTransfer.files;
  if (files.length) {
    handleFiles(files[0]);
  }
}

function onFileSelect(e) {
  const files = e.target.files;
  if (files.length) {
    handleFiles(files[0]);
  }
}

function handleFiles(file) {
  // Check if file is an image
  if (!file.type.match('image.*')) {
    emit('error', 'Please select an image file');
    return;
  }
  
  // Check file size (limit to 10MB)
  if (file.size > 10 * 1024 * 1024) {
    emit('error', 'File size should be less than 10MB');
    return;
  }
  
  // Emit file to parent
  emit('update:modelValue', file);
  
  // Create preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.value = e.target.result;
  };
  reader.readAsDataURL(file);
}

function removeImage(e) {
  e.preventDefault();
  previewImage.value = null;
  fileInput.value.value = '';
  emit('update:modelValue', null);
}
</script>

<style scoped>
.dropzone {
  border: 2px dashed #D1D5DB;
  border-radius: 12px;
  padding: 2.5rem;
  transition: all 0.3s ease;
  background: #F9FAFB;
  cursor: pointer;
  margin-bottom: 1.5rem;
  overflow: hidden;
}

.dropzone:hover {
  border-color: #6366F1;
  background: rgba(99, 102, 241, 0.05);
}

.dropzone.active {
  border-color: #6366F1;
  background: rgba(99, 102, 241, 0.05);
  transform: scale(1.01);
}

.dropzone.has-image {
  padding: 0;
  overflow: hidden;
}

.file-input {
  display: none;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.upload-icon {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: rgba(99, 102, 241, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  color: #6366F1;
}

.upload-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1F2937;
}

.upload-desc {
  font-size: 1rem;
  color: #6B7280;
  margin-bottom: 0.5rem;
}

.upload-formats {
  font-size: 0.9rem;
  color: #9CA3AF;
}

.preview-container {
  position: relative;
  width: 100%;
  height: 0;
  padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
  overflow: hidden;
}

.preview-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 10px;
}

.remove-preview {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(255, 255, 255, 0.9);
  border: none;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: #EF4444;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.remove-preview:hover {
  background: white;
  transform: scale(1.05);
}
</style>
