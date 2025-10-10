# API Conversion Roadmap

## ✅ Completed Endpoints

1. **POST /anthropic** - Chat completion with Claude models
   - Request: `prompt`, `max_tokens`, `max_tool_uses`, `model`
   - Response: AI response text with metadata

2. **POST /gpt** - Chat with GPT-4 models
   - Request: `prompt`, `seed`, `model_name`
   - Response: AI response text with metadata

3. **POST /flux** - Generate images with FLUX models
   - Request: `prompt`, `model`, `image_size`, `guidance_scale`
   - Response: Image URL with generation parameters

4. **POST /say** - Text-to-speech generation
   - Request: Form data with `text`, `voice`, `speed`
   - Response: MP4 file with audio waveform video

5. **POST /rembg** - Background removal (partial)
   - Request: Image file upload
   - Response: Currently returns 501 (needs URL hosting solution)

## 🚧 Endpoints To Implement

### 1. **POST /youtube** - YouTube search
   - Input: `query` string
   - Output: Top video result with URL and metadata
   - Requires: YOUTUBE_API_KEY

### 2. **GET /temp** - Weather temperature
   - Input: None (hardcoded to Fayetteville, AR)
   - Output: Current temperature and conditions
   - Requires: NWS API integration

### 3. **POST /google** - Google search
   - Input: `query` string
   - Output: Search results array
   - Requires: GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID

### 4. **POST /sd3_5_large** - Stable Diffusion 3.5 Large
   - Input: `prompt`, `model`, `guidance_scale`
   - Output: Image URL
   - Note: Can reuse flux endpoint pattern

### 5. **POST /o1** - OpenAI O1 models
   - Input: `prompt`, `model_name`, `seed`
   - Output: AI response text
   - Note: Similar to GPT endpoint

### 6. **POST /t2v** - Text to video
   - Input: `text`, `length`, `steps`, `seed`
   - Output: Video file
   - Requires: ComfyUI integration

### 7. **POST /gptimg/generate** - Generate images with GPT
   - Input: `prompt`, `model`, `size`, `quality`, `transparent_background`
   - Output: PNG image file

### 8. **POST /gptimg/edit** - Edit images with GPT
   - Input: Multiple image uploads + prompt + parameters
   - Output: Edited PNG image file
   - Note: Most complex endpoint with progress tracking

## 📝 Implementation Notes

### File Handling
- Discord attachments → FastAPI UploadFile
- Discord file responses → FileResponse or URL responses
- Progress tracking → WebSocket or SSE (for gptimg)

### Error Handling
- Keep consistent 500 status for service errors
- Add 400 for validation errors
- Consider rate limiting

### Testing
- Basic tests created for all endpoints
- Implemented endpoints have working tests
- Placeholder tests for unimplemented endpoints

## 🚀 Next Steps

1. Implement remaining endpoints in priority order
2. Add authentication/API keys
3. Add rate limiting
4. Deploy with Docker/Fly.io
5. Add OpenAPI documentation customization
6. Consider WebSocket support for long-running operations