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

## ✅ Additional Completed Endpoints

6. **POST /youtube** - YouTube search
   - Request: `query` string
   - Response: Top video URL and metadata

7. **GET /temp** - Weather temperature
   - Response: Current temperature for Fayetteville, AR

8. **POST /google** - Google search
   - Request: `query` string
   - Response: Search results array

9. **POST /o1** - OpenAI O1 models
   - Request: `prompt`, `model_name`, `seed`
   - Response: AI response text

10. **POST /t2v** - Text to video
    - Request: `text`, `length`, `steps`, `seed`
    - Response: Video file (requires ComfyUI)

11. **POST /gptimg/generate** - Generate images with GPT
    - Request: `prompt`, `model`, `size`, `quality`, `transparent_background`
    - Response: PNG image file

12. **POST /gptimg/edit** - Edit images with GPT
    - Request: Multiple image uploads + prompt + parameters
    - Response: Edited PNG image file

## 🚀 All Endpoints Implemented!

All Discord slash commands have been successfully converted to REST API endpoints.

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