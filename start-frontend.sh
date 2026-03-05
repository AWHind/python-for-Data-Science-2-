#!/bin/bash

# Script to start the frontend development server

echo "Starting CardioSense Frontend..."
echo "================================"

cd front-end

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

echo "Starting Next.js dev server on port 3000..."
npm run dev -- --port 3000 --hostname 0.0.0.0
