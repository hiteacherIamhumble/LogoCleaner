<template>
  <div class="processing-status">
    <div class="progress-tracker">
      <div 
        v-for="(step, index) in steps" 
        :key="index" 
        class="progress-step"
        :class="{
          'active': currentStep >= index,
          'completed': currentStep > index
        }"
      >
        <div class="step-indicator">
          <div class="step-icon">
            <svg v-if="currentStep > index" class="check-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
            </svg>
            <span v-else>{{ index + 1 }}</span>
          </div>
          <div class="step-line" v-if="index < steps.length - 1"></div>
        </div>
        <div class="step-details">
          <h3 class="step-title">{{ step.title }}</h3>
          <p class="step-description">{{ step.description }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue';

defineProps({
  steps: {
    type: Array,
    required: true
  },
  currentStep: {
    type: Number,
    default: 0
  }
});
</script>

<style scoped>
.processing-status {
  margin: 2rem 0;
}

.progress-tracker {
  max-width: 800px;
  margin: 0 auto;
}

.progress-step {
  display: flex;
  margin-bottom: 2rem;
  opacity: 0.5;
  transition: opacity 0.5s ease;
}

.progress-step.active {
  opacity: 1;
}

.step-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-right: 1.5rem;
}

.step-icon {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: white;
  border: 2px solid #D1D5DB;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  color: #6B7280;
  margin-bottom: 0.5rem;
  transition: all 0.3s ease;
}

.active .step-icon {
  border-color: #6366F1;
  color: #6366F1;
}

.completed .step-icon {
  background: #6366F1;
  border-color: #6366F1;
  color: white;
}

.check-icon {
  width: 24px;
  height: 24px;
  fill: white;
}

.step-line {
  flex-grow: 1;
  width: 2px;
  background: #D1D5DB;
  height: 100%;
  transition: background 0.3s ease;
}

.active .step-line {
  background: #6366F1;
}

.step-details {
  flex: 1;
}

.step-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1F2937;
  transition: color 0.3s ease;
}

.active .step-title {
  color: #6366F1;
}

.step-description {
  color: #6B7280;
  line-height: 1.5;
}

@media (max-width: 768px) {
  .step-icon {
    width: 40px;
    height: 40px;
  }
  
  .step-title {
    font-size: 1.1rem;
  }
}
</style>
