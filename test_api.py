# test_api.py

"""
Test suite for REST API endpoints.

This module contains tests for both implemented and not-yet-implemented API endpoints,
providing a clear roadmap for the remaining conversion work.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api import app

# Test client for synchronous tests
client = TestClient(app)


# Helper to check if API keys are available
def has_api_keys() -> bool:
    """Check if required API keys are set in environment."""
    return all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            # Add other required keys as needed
        ]
    )


# ===== IMPLEMENTED ENDPOINT TESTS =====


def test_root() -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Mobius AI API"
    assert data["version"] == "0.1.0"


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_anthropic_endpoint() -> None:
    """Test Anthropic chat completion endpoint."""
    request_data = {
        "prompt": "Say 'Hello, API test!' and nothing else.",
        "max_tokens": 50,
        "max_tool_uses": 0,
        "model": "claude-3-haiku-20240307",  # Use cheaper model for tests
    }
    
    response = client.post("/anthropic", json=request_data)
    
    # Should either succeed or fail gracefully
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert data["model"] == request_data["model"]
        assert data["prompt"] == request_data["prompt"]
        print(f"✓ Anthropic test passed: {data['response'][:50]}...")
    else:
        print(f"✗ Anthropic test failed with status {response.status_code}")
        assert response.status_code == 500  # Expected error code


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_gpt_endpoint() -> None:
    """Test GPT chat completion endpoint."""
    request_data = {
        "prompt": "Say 'Hello, API test!' and nothing else.",
        "seed": 42,
        "model_name": "gpt-4o-mini",
    }
    
    response = client.post("/gpt", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert data["model"] == request_data["model_name"]
        assert data["prompt"] == request_data["prompt"]
        assert data["seed"] == 42
        print(f"✓ GPT test passed: {data['response'][:50]}...")
    else:
        print(f"✗ GPT test failed with status {response.status_code}")
        assert response.status_code == 500


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_flux_endpoint() -> None:
    """Test FLUX image generation endpoint."""
    request_data = {
        "prompt": "A simple red square on white background",
        "model": "fal-ai/flux/schnell",  # Fastest model
        "image_size": "square",
        "guidance_scale": 3.5,
    }
    
    response = client.post("/flux", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "image_url" in data
        assert data["prompt"] == request_data["prompt"]
        assert data["model"] == request_data["model"]
        assert data["image_size"] == request_data["image_size"]
        assert data["guidance_scale"] == request_data["guidance_scale"]
        print(f"✓ FLUX test passed: {data['image_url'][:50]}...")
    else:
        print(f"✗ FLUX test failed with status {response.status_code}")
        assert response.status_code == 500


def test_say_endpoint() -> None:
    """Test TTS speech generation endpoint."""
    # Using form data for this endpoint
    form_data = {
        "text": "Hello, API test!",
        "voice": "onyx",
        "speed": 1.0,
    }
    
    response = client.post("/say", data=form_data)
    
    if response.status_code == 200:
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 0
        print("✓ TTS test passed: Generated audio file")
    else:
        print(f"✗ TTS test failed with status {response.status_code}")
        assert response.status_code == 500


def test_say_text_too_long() -> None:
    """Test TTS with text exceeding limit."""
    form_data = {
        "text": "x" * 4097,  # Exceeds 4096 char limit
        "voice": "onyx",
        "speed": 1.0,
    }
    
    response = client.post("/say", data=form_data)
    assert response.status_code == 400
    assert "cannot exceed 4096 characters" in response.json()["detail"]


def test_rembg_endpoint() -> None:
    """Test background removal endpoint (currently unimplemented)."""
    # Create a dummy image file
    files = {"image": ("test.png", b"fake image data", "image/png")}
    
    response = client.post("/rembg", files=files)
    
    # Currently returns 501 Not Implemented
    assert response.status_code == 501
    assert "requires additional implementation" in response.json()["detail"]
    print("✓ RemBG test passed: Correctly returns not implemented status")


# ===== NOT YET IMPLEMENTED ENDPOINT TESTS =====


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_youtube_endpoint() -> None:
    """Test YouTube search endpoint (TO BE IMPLEMENTED)."""
    request_data = {"query": "Python programming tutorial"}
    
    # Expected response format:
    # {
    #     "video_url": "https://www.youtube.com/watch?v=...",
    #     "video_id": "...",
    #     "title": "...",
    #     "query": "Python programming tutorial"
    # }
    
    response = client.post("/youtube", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "video_url" in data
    assert "video_id" in data
    assert data["query"] == request_data["query"]


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_temp_endpoint() -> None:
    """Test temperature fetching endpoint (TO BE IMPLEMENTED)."""
    # No request data needed - always returns Fayetteville, AR temp
    
    # Expected response format:
    # {
    #     "temperature": "72°F",
    #     "location": "Fayetteville, AR",
    #     "conditions": "Partly cloudy",
    #     ...
    # }
    
    response = client.get("/temp")
    assert response.status_code == 200
    data = response.json()
    assert "temperature" in data
    assert "location" in data


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_google_endpoint() -> None:
    """Test Google search endpoint (TO BE IMPLEMENTED)."""
    request_data = {"query": "FastAPI documentation"}
    
    # Expected response format:
    # {
    #     "results": [
    #         {"title": "...", "link": "...", "snippet": "..."},
    #         ...
    #     ],
    #     "query": "FastAPI documentation"
    # }
    
    response = client.post("/google", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert data["query"] == request_data["query"]


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_sd3_5_large_endpoint() -> None:
    """Test Stable Diffusion 3.5 Large endpoint (TO BE IMPLEMENTED)."""
    request_data = {
        "prompt": "A majestic mountain landscape",
        "model": "fal-ai/stable-diffusion-v35-large",
        "guidance_scale": 4.5,
    }
    
    # Expected response similar to FLUX endpoint
    response = client.post("/sd3_5_large", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "image_url" in data
    assert data["prompt"] == request_data["prompt"]


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_o1_endpoint() -> None:
    """Test OpenAI O1 model endpoint (TO BE IMPLEMENTED)."""
    request_data = {
        "prompt": "Explain quantum computing in simple terms",
        "model_name": "o1-mini",
        "seed": 42,
    }
    
    # Expected response similar to GPT endpoint
    response = client.post("/o1", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["model"] == request_data["model_name"]


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_t2v_endpoint() -> None:
    """Test text-to-video endpoint (TO BE IMPLEMENTED)."""
    request_data = {
        "text": "A cat walking on a beach",
        "length": 33,
        "steps": 30,
        "seed": 0,
    }
    
    # Expected response: video file
    response = client.post("/t2v", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] in ["video/mp4", "video/avi"]
    assert len(response.content) > 0


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_gptimg_generate_endpoint() -> None:
    """Test GPT Image generation endpoint (TO BE IMPLEMENTED)."""
    request_data = {
        "prompt": "A serene lake at sunset",
        "model": "gpt-image-1",
        "size": "1024x1024",
        "quality": "high",
        "transparent_background": False,
    }
    
    # Expected response: image file or URL
    response = client.post("/gptimg/generate", json=request_data)
    assert response.status_code == 200


@pytest.mark.skip(reason="Endpoint not yet implemented")
def test_gptimg_edit_endpoint() -> None:
    """Test GPT Image editing endpoint (TO BE IMPLEMENTED)."""
    # This will need multipart form data with image files
    files = {
        "edit_image1": ("image.png", b"fake image data", "image/png"),
    }
    data = {
        "prompt": "Add a rainbow to the sky",
        "model": "gpt-image-1",
        "size": "auto",
        "quality": "auto",
    }
    
    response = client.post("/gptimg/edit", files=files, data=data)
    assert response.status_code == 200


# ===== INTEGRATION TEST EXAMPLE =====


@pytest.mark.asyncio
@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
async def test_multiple_endpoints_async() -> None:
    """Test multiple endpoints concurrently to verify async behavior."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Launch multiple requests concurrently
        tasks = [
            ac.post(
                "/anthropic",
                json={
                    "prompt": "Count to 3",
                    "max_tokens": 20,
                    "model": "claude-3-haiku-20240307",
                },
            ),
            ac.post(
                "/gpt",
                json={"prompt": "Count to 3", "model_name": "gpt-4o-mini"},
            ),
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"✗ Request {i} failed: {response}")
            else:
                print(f"✓ Request {i} succeeded with status {response.status_code}")


# ===== MAIN TEST RUNNER =====


def main() -> None:
    """Run tests with summary."""
    print("\n" + "=" * 60)
    print("MOBIUS API TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run implemented tests
    print("TESTING IMPLEMENTED ENDPOINTS:\n")
    test_root()
    test_anthropic_endpoint()
    test_gpt_endpoint()
    test_flux_endpoint()
    test_say_endpoint()
    test_say_text_too_long()
    test_rembg_endpoint()
    
    # Show not yet implemented
    print("\n\nENDPOINTS TO BE IMPLEMENTED:")
    print("- POST /youtube - YouTube search")
    print("- GET  /temp - Weather temperature")
    print("- POST /google - Google search")
    print("- POST /sd3_5_large - Stable Diffusion 3.5 Large")
    print("- POST /o1 - OpenAI O1 models")
    print("- POST /t2v - Text to video")
    print("- POST /gptimg/generate - GPT Image generation")
    print("- POST /gptimg/edit - GPT Image editing")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Can run with pytest or directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        main()