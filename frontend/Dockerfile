FROM node:16-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (creating these files if they don't exist)
RUN if [ -f package.json ]; then npm install; else echo '{"name":"clipsummary-frontend","version":"1.0.0","scripts":{"start":"http-server -p 3000 --cors -c-1"}}' > package.json && npm install http-server --save-dev; fi

# Copy frontend files
COPY . .

# Expose port
EXPOSE 3000

# Command to run development server
CMD ["npm", "start"]