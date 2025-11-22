# FIXED: Upgrade to Node 20-alpine (required by newer Next.js versions)
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files first to leverage caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose port 3000
EXPOSE 3000

# Start the application
CMD ["npm", "start"]