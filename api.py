# api.py

"""
REST API endpoints for AI services.

This module provides FastAPI REST endpoints that expose the same functionality
as the Discord bot slash commands, allowing direct API access to AI services.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import services

# Load environment variables from .env file
load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mobius AI API",
    description="REST API for AI services including chat completions, image generation, and more",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Type aliases matching commands.py
AnthropicModel = Literal[
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-1",
]
GPTModel = Literal["gpt-5", "gpt-5-mini", "gpt-5-nano"]
GPTImageModel = Literal["gpt-image-1"]
GPTImageSize = Literal["auto", "1024x1024", "1536x1024", "1024x1536"]
GPTImageQuality = Literal["auto", "low", "medium", "high"]


# Request/Response models
class AnthropicRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for the AI model")
    max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens to generate")
    model: AnthropicModel = Field(default="claude-sonnet-4-5", description="Anthropic model to use")


class AnthropicResponse(BaseModel):
    response: str = Field(..., description="The AI model's response")
    model: str = Field(..., description="Model used for generation")
    prompt: str = Field(..., description="Original prompt")


class GPTRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for the AI model")
    seed: int | None = Field(default=None, description="Seed for reproducible generation")
    model_name: GPTModel = Field(default="gpt-5-mini", description="GPT model to use")


class GPTResponse(BaseModel):
    response: str = Field(..., description="The AI model's response")
    model: str = Field(..., description="Model used for generation")
    prompt: str = Field(..., description="Original prompt")
    seed: int | None = Field(default=None, description="Seed used for generation")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {"message": "Mobius AI API", "version": "0.1.0"}


@app.post("/anthropic", response_model=AnthropicResponse)
async def anthropic_chat(request: AnthropicRequest) -> AnthropicResponse:
    """Chat completion with Anthropic Claude models."""
    try:
        response_text: str = await services.anthropic_chat_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            model=request.model,
        )

        return AnthropicResponse(
            response=response_text,
            model=request.model,
            prompt=request.prompt,
        )
    except Exception as e:
        logger.exception(f"Error with Anthropic API for prompt: {request.prompt}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while communicating with the Anthropic API",
        ) from e


@app.post("/gpt", response_model=GPTResponse)
async def gpt_chat(request: GPTRequest) -> GPTResponse:
    """Chat with OpenAI's GPT models."""
    try:
        # Convert to the message format expected by services.gpt_chat_completion
        history: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": request.prompt}]},
        ]

        # Handle seed conversion
        api_seed: int | None = int(request.seed) if request.seed is not None and request.seed != -1 else None

        response_text: str = await services.gpt_chat_completion(history, request.model_name, api_seed)

        return GPTResponse(
            response=response_text,
            model=request.model_name,
            prompt=request.prompt,
            seed=api_seed,
        )
    except Exception as e:
        logger.exception(f"Error with GPT API for prompt: {request.prompt}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while communicating with the OpenAI API",
        ) from e


@app.post("/say")
async def generate_speech(
    text: str = Form(..., description="Text to convert to speech (max 4096 chars)"),
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Form(default="onyx"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
) -> FileResponse:
    """Generate speech from text using OpenAI's TTS API."""
    if len(text) > 4096:
        raise HTTPException(status_code=400, detail="Text cannot exceed 4096 characters")

    try:
        waveform_video_file_path: Path = await services.generate_speech(text, voice, speed)

        return FileResponse(
            path=waveform_video_file_path,
            filename=waveform_video_file_path.name,
            media_type="video/mp4",
        )
    except Exception as e:
        logger.exception(f"Error generating speech for text: {text}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating speech",
        ) from e


@app.post("/rembg")
async def remove_background(
    image: UploadFile = File(..., description="Image file to remove background from"),
) -> dict[str, str]:
    """Remove background from an image using fal.ai/imageutils/rembg."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file (PNG, JPG, WEBP)",
        )

    try:
        # Ensure proper file extension based on content type
        if image.content_type == "image/jpeg":
            suffix = ".jpg"
        elif image.content_type == "image/png":
            suffix = ".png"
        elif image.content_type == "image/webp":
            suffix = ".webp"
        else:
            # Fallback to original filename extension
            suffix = Path(image.filename or "image.png").suffix

        # Save uploaded file temporarily to get a URL
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        # Note: This would need modification to work with fal-ai
        # as it expects a URL, not a local file path
        # For now, returning an error indicating this limitation
        tmp_path.unlink()  # Clean up

        raise HTTPException(
            status_code=501,
            detail="Background removal API endpoint requires additional implementation for file URL hosting",
        )

    except HTTPException:
        # Re-raise HTTPException without wrapping it
        raise
    except Exception as e:
        logger.exception(f"Error removing background from image: {image.filename}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while removing the image background",
        ) from e


# Additional Request/Response models for remaining endpoints
class YouTubeRequest(BaseModel):
    query: str = Field(..., description="YouTube search query")


class YouTubeResponse(BaseModel):
    video_url: str = Field(..., description="URL of the top video result")
    video_id: str = Field(..., description="YouTube video ID")
    title: str | None = Field(default=None, description="Video title")
    query: str = Field(..., description="Original search query")


class GoogleRequest(BaseModel):
    query: str = Field(..., description="Google search query")


class GoogleResult(BaseModel):
    title: str = Field(..., description="Result title")
    link: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet")


class GoogleResponse(BaseModel):
    results: list[GoogleResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")




class GPTImageGenerateRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    model: GPTImageModel = Field(default="gpt-image-1", description="GPT Image model to use")
    size: GPTImageSize = Field(default="auto", description="Image size")
    quality: GPTImageQuality = Field(default="auto", description="Image quality")
    transparent_background: bool = Field(default=False, description="Enable transparent background")


class TemperatureResponse(BaseModel):
    temperature: str = Field(..., description="Current temperature")
    location: str = Field(default="Fayetteville, AR", description="Location")
    conditions: str | None = Field(default=None, description="Weather conditions")
    full_report: str = Field(..., description="Full weather report text")


@app.post("/youtube", response_model=YouTubeResponse)
async def youtube_search(request: YouTubeRequest) -> YouTubeResponse:
    """Search YouTube and return the top video result."""
    import os

    youtube_api_key: str | None = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        logger.error("YOUTUBE_API_KEY environment variable not set.")
        raise HTTPException(
            status_code=503,
            detail="YouTube API key is not configured. This endpoint is unavailable.",
        )

    try:
        result: dict[str, Any] = await services.get_top_youtube_result(request.query, youtube_api_key)

        if "error" in result:
            raise HTTPException(status_code=400, detail=f"YouTube search error: {result['error']}")

        if "videoId" not in result:
            raise HTTPException(status_code=404, detail=f"No results found for '{request.query}'")

        video_url: str = f"https://www.youtube.com/watch?v={result['videoId']}"

        return YouTubeResponse(
            video_url=video_url,
            video_id=result["videoId"],
            title=result.get("title"),
            query=request.query,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during YouTube search for query: {request.query}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during the YouTube search",
        ) from e


@app.get("/temp", response_model=TemperatureResponse)
async def get_temperature() -> TemperatureResponse:
    """Get the current temperature in Fayetteville, AR."""
    try:
        temperature_info: str = await services.temp_callback()

        # Parse the temperature info to extract structured data
        # The service returns a formatted string, so we need to parse it
        lines = temperature_info.strip().split("\n")
        temp_line = next((line for line in lines if "°F" in line), "")
        conditions_line = next((line for line in lines if "Conditions:" in line), "")

        # Extract temperature value
        temp_match = temp_line.split(":")[-1].strip() if temp_line else "N/A"
        conditions_match = conditions_line.split(":")[-1].strip() if conditions_line else None

        return TemperatureResponse(
            temperature=temp_match,
            location="Fayetteville, AR",
            conditions=conditions_match,
            full_report=temperature_info,
        )
    except Exception as e:
        logger.exception("Error fetching temperature data")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching temperature data",
        ) from e


@app.post("/google", response_model=GoogleResponse)
async def google_search(request: GoogleRequest) -> GoogleResponse:
    """Search the web using Google Custom Search API."""
    import os

    if not os.getenv("GOOGLE_SEARCH_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        logger.error("Google Search API key or CSE ID not configured.")
        raise HTTPException(
            status_code=503,
            detail="Google Search is not configured. This endpoint is unavailable.",
        )

    try:
        search_results: str = await services.google_search(request.query)

        if not search_results:
            return GoogleResponse(results=[], query=request.query)

        # Parse the formatted string response into structured data
        results: list[GoogleResult] = []
        lines = search_results.strip().split("\n")

        for i in range(0, len(lines), 3):  # Results are typically formatted in groups
            if i + 1 < len(lines) and lines[i].startswith("**") and lines[i].endswith("**"):
                title = lines[i].strip("*").strip()
                link = lines[i + 1].strip() if i + 1 < len(lines) else ""
                snippet = lines[i + 2].strip() if i + 2 < len(lines) else ""

                if link.startswith("http"):
                    results.append(
                        GoogleResult(
                            title=title,
                            link=link,
                            snippet=snippet,
                        )
                    )

        return GoogleResponse(results=results, query=request.query)
    except Exception as e:
        logger.exception(f"Error during Google search for query: {request.query}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during the Google search",
        ) from e

@app.post("/gptimg/generate")
async def gptimg_generate(request: GPTImageGenerateRequest) -> FileResponse:
    """Generate images using OpenAI's GPT Image model."""
    try:
        generated_image_path: Path = await services.generate_gpt_image(
            prompt=request.prompt,
            model=request.model,
            quality=request.quality,
            size=request.size,
            transparent_background=request.transparent_background,
        )

        return FileResponse(
            path=generated_image_path,
            filename=generated_image_path.name,
            media_type="image/png",
        )
    except Exception as e:
        logger.exception(f"Error generating GPT image for prompt: {request.prompt}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the image",
        ) from e


@app.post("/gptimg/edit")
async def gptimg_edit(
    prompt: str = Form(..., description="Edit instructions"),
    model: GPTImageModel = Form(default="gpt-image-1"),
    size: GPTImageSize = Form(default="auto"),
    edit_images: list[UploadFile] = File(..., description="Images to edit (up to 5)"),
    mask_image: UploadFile | None = File(default=None, description="Optional mask image"),
) -> FileResponse:
    """Edit images using OpenAI's GPT Image model."""
    if len(edit_images) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed for editing")

    if not edit_images:
        raise HTTPException(status_code=400, detail="At least one image required for editing")

    temp_image_paths: list[Path] = []
    temp_mask_path: Path | None = None

    try:
        # Save uploaded images
        for img in edit_images:
            if not img.content_type or not img.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {img.filename} is not a valid image type",
                )

            # Ensure proper file extension based on content type
            if img.content_type == "image/jpeg":
                suffix = ".jpg"
            elif img.content_type == "image/png":
                suffix = ".png"
            elif img.content_type == "image/webp":
                suffix = ".webp"
            else:
                # Fallback to original filename extension
                suffix = Path(img.filename or "image.png").suffix

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                content = await img.read()
                tmp_file.write(content)
                temp_image_paths.append(Path(tmp_file.name))

        # Save mask if provided
        if mask_image:
            if not mask_image.content_type or not mask_image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="Mask file is not a valid image type",
                )

            # Ensure proper file extension for mask based on content type
            if mask_image.content_type == "image/jpeg":
                mask_suffix = ".jpg"
            elif mask_image.content_type == "image/png":
                mask_suffix = ".png"
            elif mask_image.content_type == "image/webp":
                mask_suffix = ".webp"
            else:
                # Fallback to original filename extension
                mask_suffix = Path(mask_image.filename or "mask.png").suffix

            with tempfile.NamedTemporaryFile(suffix=mask_suffix, delete=False) as tmp_mask_file:
                mask_content = await mask_image.read()
                tmp_mask_file.write(mask_content)
                temp_mask_path = Path(tmp_mask_file.name)

        # Call the service
        edited_image_path: Path = await services.edit_gpt_image(
            prompt=prompt,
            images=temp_image_paths,
            mask=temp_mask_path,
            model=model,
            size=size,
        )

        return FileResponse(
            path=edited_image_path,
            filename=edited_image_path.name,
            media_type="image/png",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error editing GPT image for prompt: {prompt}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while editing the image",
        ) from e
    finally:
        # Clean up temporary files
        for p in temp_image_paths:
            if p.exists():
                try:
                    p.unlink()
                except OSError as e_unlink:
                    logger.exception(f"Error deleting temp file {p}: {e_unlink}")

        if temp_mask_path and temp_mask_path.exists():
            try:
                temp_mask_path.unlink()
            except OSError as e_unlink:
                logger.exception(f"Error deleting temp mask file: {e_unlink}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
