# Qwen3-Vision API

A FastAPI-based computer vision service using Qwen2-VL-2B-Instruct for detailed video and image analysis. Provides comprehensive descriptions of visual content through two specialized endpoints.

## Features

- 🎬 **Video Analysis**: Upload videos and get extremely detailed scene descriptions
- 🖼️ **Batch Image Analysis**: Process up to 60 images simultaneously with detailed descriptions
- 🔒 **API Key Authentication**: Secure access with custom API key
- 🚀 **High Quality**: Uses Qwen2-VL-2B-Instruct for accurate visual understanding
- ☁️ **Railway Ready**: Optimized for Railway deployment with pre-downloaded models

## Supported Formats

**Videos**: MP4, AVI, MOV, WebM, QuickTime
**Images**: JPG, JPEG, PNG, WebP, GIF

**File Size Limit**: 100MB per file
**Image Batch Limit**: 60 images maximum

## Quick Start

### 1. Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd qwen3-vision-api

# Build and run with Docker
docker build -t qwen-vision-api .
docker run -p 8000:8000 -e API_KEY=your_secret_key qwen-vision-api
```

### 2. Test the API

**Health Check:**

```bash
curl http://localhost:8000/
```

**Analyze Video:**

```bash
curl -X POST "http://localhost:8000/analyze-video" \
  -H "X-API-Key: your_secret_key" \
  -F "video_file=@your_video.mp4"
```

**Analyze Images:**

```bash
curl -X POST "http://localhost:8000/analyze-images" \
  -H "X-API-Key: your_secret_key" \
  -F "image_files=@image1.jpg" \
  -F "image_files=@image2.jpg"
```

## Environment Variables

Create a `.env` file or set these in your deployment:

```bash
API_KEY=your_secret_api_key_here
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
PORT=8000
```

## API Endpoints

### `GET /`

Health check endpoint

### `POST /analyze-video`

Analyze single video file and return comprehensive description

**Headers:**

- `X-API-Key`: Your API key (required)

**Body:**

- `video_file`: Video file (multipart/form-data)

**Response:**

```json
{
  "success": true,
  "analysis": {
    "filename": "video.mp4",
    "description": "Extremely detailed video description covering scenes, people, actions, visual elements, atmosphere, and more...",
    "duration": 30.5,
    "key_moments": ["Timestamps of important events"]
  }
}
```

### `POST /analyze-images`

Analyze up to 60 images in batch

**Headers:**

- `X-API-Key`: Your API key (required)

**Body:**

- `image_files`: Multiple image files (multipart/form-data)

**Response:**

```json
{
  "success": true,
  "total_images": 3,
  "analyses": [
    {
      "image_id": "img_001",
      "filename": "photo.jpg",
      "description": "Extremely detailed description of main subjects, setting, colors, lighting, composition, mood, and all visual elements...",
      "confidence": null
    }
  ]
}
```

## Deploy to Railway

1. Push this repo to GitHub
2. Connect GitHub repo to Railway
3. Add environment variable: `API_KEY=your_secret_key`
4. Deploy!

Railway will automatically:

- Build using the Dockerfile
- Download the Qwen2-VL-2B model during build
- Install video/image processing dependencies
- Start the API server

## Technical Details

- **Framework**: FastAPI
- **Model**: Qwen2-VL-2B-Instruct (~4GB)
- **Video Processing**: OpenCV + FFmpeg for frame extraction
- **Image Processing**: Pillow + OpenCV
- **Memory Usage**: ~6-8GB RAM recommended
- **Processing Speed**:
  - Videos: ~30-60 seconds depending on length
  - Images: ~2-3 seconds per image

## Analysis Features

### Video Analysis Includes:

- Scene and setting description
- People and objects identification
- Chronological action sequence
- Visual elements (colors, composition, lighting)
- Atmosphere and mood assessment
- Text and graphics detection
- Audio-visual correlation insights

### Image Analysis Includes:

- Main subjects detailed description
- Setting and environment analysis
- Color and lighting assessment
- Composition and style evaluation
- Texture and detail identification
- Mood and atmosphere capture
- Text and graphics recognition
- Movement and action implications

## Error Codes

- `400`: Invalid file type or too many images
- `401`: Invalid or missing API key
- `413`: File too large (>100MB)
- `500`: Analysis processing error

## Interactive Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation with upload interfaces.

## License

MIT License
