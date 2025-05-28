# Use an official Node runtime as a parent image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install project dependencies
RUN npm install
# If you are using yarn:
# RUN yarn install

# Bundle app source
COPY . .

# Make port 5173 available to the world outside this container
EXPOSE 5173

# Define the command to run your app
CMD [ "npm", "run", "dev" ]
# Or your specific start command, e.g., "node", "server.js"