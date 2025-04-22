# LogoCleaner Frontend

This is the Vue.js frontend for the LogoCleaner application. It's built with Vue 2 and uses Vite as the build tool.

## Setup Instructions

### Prerequisites
- Node.js (v14+)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
# or
yarn
```

### Development

```bash
# Start development server
npm run dev
# or
yarn dev
```

This will start the development server at http://localhost:8080.

### Building for Production

```bash
# Build for production
npm run build
# or
yarn build
```

This will generate a `dist` directory with the built application.

### Preview Production Build

```bash
# Preview production build
npm run preview
# or
yarn preview
```

## Configuration

The application is configured to connect to a backend server running at http://localhost:5050. This can be changed in the `.env` file:

```
VUE_APP_API_URL=http://localhost:5050
```

## Project Structure

- `src/components/`: Reusable Vue components
- `src/views/`: Page components
- `src/assets/`: Static assets (images, etc.)
- `src/api/`: API service layer
- `src/router/`: Vue Router configuration

## Troubleshooting

If you encounter the "module does not provide an export named 'default'" error, it's likely due to Vue 3 vs Vue 2 compatibility issues. Make sure your package.json has the correct dependencies for Vue 2:

```json
"dependencies": {
  "vue": "^2.6.14",
  "vue-router": "^3.6.5"
}
```

And that you're using the correct plugin in vite.config.js:

```js
import { createVuePlugin as vue2 } from '@vitejs/plugin-vue2'
```
