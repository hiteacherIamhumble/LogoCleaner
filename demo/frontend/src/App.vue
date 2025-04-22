<template>
  <div id="app">
    <div class="navbar">
      <div class="logo">
        <div class="logo-icon">
          <span class="logo-text">LC</span>
        </div>
        <h1 class="brand-name">LogoCleaner</h1>
      </div>
      <div class="nav-links">
        <router-link to="/" class="nav-link">Home</router-link>
        <a href="#features" class="nav-link">Features</a>
        <a href="#examples" class="nav-link">Examples</a>
      </div>
    </div>
    
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
    
    <footer class="footer">
      <div class="footer-content">
        <div class="footer-logo">
          <div class="logo-icon small">
            <span class="logo-text">LC</span>
          </div>
          <span class="footer-brand">LogoCleaner</span>
        </div>
        <div class="footer-links">
          <a href="#" class="footer-link">Terms</a>
          <a href="#" class="footer-link">Privacy</a>
          <a href="#" class="footer-link">Contact</a>
        </div>
        <div class="copyright">
          &copy; {{ new Date().getFullYear() }} LogoCleaner. All rights reserved.
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
// No additional script needed for this component
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  font-family: 'Poppins', sans-serif;
  background: #ffffff;
  color: #333;
  overflow-x: hidden;
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 5%;
  background: #fff;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 1000;
}

.logo {
  display: flex;
  align-items: center;
}

.logo-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px;
  box-shadow: 0 4px 10px rgba(107, 114, 241, 0.3);
}

.logo-icon.small {
  width: 30px;
  height: 30px;
}

.logo-text {
  color: white;
  font-weight: 700;
  font-size: 16px;
}

.brand-name {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  text-decoration: none;
  color: #4B5563;
  font-weight: 500;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  position: relative;
}

.nav-link::after {
  content: '';
  position: absolute;
  bottom: -5px;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  transition: width 0.3s ease;
}

.nav-link:hover {
  color: #6366F1;
}

.nav-link:hover::after,
.router-link-active::after {
  width: 100%;
}

.router-link-active {
  color: #6366F1;
  font-weight: 600;
}

.footer {
  background: #F9FAFB;
  padding: 3rem 5%;
  margin-top: auto;
}

.footer-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}

.footer-logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.footer-brand {
  font-weight: 600;
  color: #4B5563;
}

.footer-links {
  display: flex;
  gap: 2rem;
}

.footer-link {
  color: #6B7280;
  text-decoration: none;
  font-size: 0.9rem;
  transition: color 0.3s ease;
}

.footer-link:hover {
  color: #6366F1;
}

.copyright {
  color: #9CA3AF;
  font-size: 0.85rem;
  text-align: center;
}

/* Transition styles */
.fade-enter-active, 
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from, 
.fade-leave-to {
  opacity: 0;
}

/* Utility classes */
.container {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.btn {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  transition: all 0.3s ease;
  border: none;
  text-align: center;
}

.btn-primary {
  background: linear-gradient(135deg, #6366F1, #8B5CF6);
  color: white;
  box-shadow: 0 4px 10px rgba(107, 114, 241, 0.3);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 15px rgba(107, 114, 241, 0.4);
}

.btn-secondary {
  background: white;
  color: #6366F1;
  border: 2px solid #6366F1;
}

.btn-secondary:hover {
  background: rgba(99, 102, 241, 0.05);
}

.btn-lg {
  padding: 1rem 2rem;
  font-size: 1.1rem;
}

.btn-block {
  display: block;
  width: 100%;
}

/* For loading animations */
@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  display: inline-block;
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255,255,255,0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease-in-out infinite;
}

@media (max-width: 768px) {
  .navbar {
    padding: 1rem 3%;
  }
  
  .nav-links {
    gap: 1rem;
  }
  
  .nav-link {
    font-size: 0.85rem;
  }
  
  .brand-name {
    font-size: 1.2rem;
  }
}
</style>
