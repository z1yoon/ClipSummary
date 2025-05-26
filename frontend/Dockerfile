FROM node:16-alpine

WORKDIR /app

# Copy all frontend files
COPY . .

# Install a simple HTTP server to serve static files
RUN npm install -g http-server

# Expose the port the server will run on
EXPOSE 3000

# Start the HTTP server
CMD ["http-server", "--port", "3000", "-a", "0.0.0.0", "--cors"]