FROM node:22-slim

RUN apt-get update && apt-get install -y \
    ffmpeg python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

# (opcjonalnie, ale polecam) doinstaluj davey na sztywno:
RUN npm install --omit=dev @snazzah/davey@^0.1.2

COPY . .
ENV NODE_ENV=production
VOLUME ["/app/recordings"]

CMD ["node", "index.js"]
