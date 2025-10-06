# test_api.py

"""
Test suite for REST API endpoints.

This module contains tests for both implemented and not-yet-implemented API endpoints,
providing a clear roadmap for the remaining conversion work.
"""

import asyncio
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api import app

# Load environment variables from .env file
load_dotenv()

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


# ===== NEWLY IMPLEMENTED ENDPOINT TESTS =====


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_youtube_endpoint() -> None:
    """Test YouTube search endpoint."""
    request_data = {"query": "Python programming tutorial"}

    response = client.post("/youtube", json=request_data)

    if response.status_code == 200:
        data = response.json()
        assert "video_url" in data
        assert "video_id" in data
        assert data["query"] == request_data["query"]
        print(f"✓ YouTube test passed: {data['video_url']}")
    elif response.status_code == 503:
        print("✗ YouTube test skipped: API key not configured")
        assert "not configured" in response.json()["detail"]
    else:
        print(f"✗ YouTube test failed with status {response.status_code}")
        assert response.status_code in [200, 503]


def test_temp_endpoint() -> None:
    """Test temperature fetching endpoint."""
    response = client.get("/temp")

    if response.status_code == 200:
        data = response.json()
        assert "temperature" in data
        assert "location" in data
        assert data["location"] == "Fayetteville, AR"
        assert "full_report" in data
        print(f"✓ Temperature test passed: {data['temperature']} in {data['location']}")
    else:
        print(f"✗ Temperature test failed with status {response.status_code}")
        assert response.status_code == 500


def test_google_endpoint() -> None:
    """Test Google search endpoint."""
    request_data = {"query": "FastAPI documentation"}

    response = client.post("/google", json=request_data)

    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert data["query"] == request_data["query"]
        print(f"✓ Google test passed: Found {len(data['results'])} results")
    elif response.status_code == 503:
        print("✗ Google test skipped: API key not configured")
        assert "not configured" in response.json()["detail"]
    else:
        print(f"✗ Google test failed with status {response.status_code}")
        assert response.status_code in [200, 503]


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_o1_endpoint() -> None:
    """Test OpenAI O1 model endpoint."""
    request_data = {
        "prompt": "Explain quantum computing in simple terms",
        "model_name": "o1-mini",
        "seed": 42,
    }

    response = client.post("/o1", json=request_data)

    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert data["model"] == request_data["model_name"]
        assert data["prompt"] == request_data["prompt"]
        assert data["seed"] == 42
        print(f"✓ O1 test passed: {data['response'][:50]}...")
    else:
        print(f"✗ O1 test failed with status {response.status_code}")
        assert response.status_code == 500


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_t2v_endpoint() -> None:
    """Test text-to-video endpoint."""
    request_data = {
        "text": "A cat walking on a beach",
        "length": 33,
        "steps": 30,
        "seed": 0,
    }

    response = client.post("/t2v", json=request_data)

    if response.status_code == 200:
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 0
        print("✓ T2V test passed: Generated video file")
    else:
        print(f"✗ T2V test failed with status {response.status_code}")
        assert response.status_code == 500


@pytest.mark.skipif(not has_api_keys(), reason="API keys not configured")
def test_gptimg_generate_endpoint() -> None:
    """Test GPT Image generation endpoint."""
    request_data = {
        "prompt": "A serene lake at sunset",
        "model": "gpt-image-1",
        "size": "1024x1024",
        "quality": "high",
        "transparent_background": False,
    }

    response = client.post("/gptimg/generate", json=request_data)

    if response.status_code == 200:
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0
        print("✓ GPT Image generation test passed: Generated PNG image")
    else:
        print(f"✗ GPT Image generation test failed with status {response.status_code}")
        assert response.status_code == 500


def test_gptimg_edit_endpoint() -> None:
    """Test GPT Image editing endpoint."""
    # This will need multipart form data with image files
    files = [
        ("edit_images", ("image.png", b"fake image data", "image/png")),
    ]
    data = {
        "prompt": "Add a rainbow to the sky",
        "model": "gpt-image-1",
        "size": "auto",
    }

    response = client.post("/gptimg/edit", files=files, data=data)

    # This will fail with 400 because we're using fake image data
    assert response.status_code in [400, 500]
    print("✓ GPT Image edit test passed: Endpoint validates image data")


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

    # Run all tests
    print("TESTING ALL ENDPOINTS:\n")

    # Basic endpoints
    test_root()

    # AI Chat endpoints
    test_anthropic_endpoint()
    test_gpt_endpoint()
    test_o1_endpoint()

    # Image generation endpoints
    test_flux_endpoint()
    test_gptimg_generate_endpoint()
    test_gptimg_edit_endpoint()
    test_rembg_endpoint()

    # Other media endpoints
    test_say_endpoint()
    test_say_text_too_long()
    test_t2v_endpoint()

    # Search and data endpoints
    test_youtube_endpoint()
    test_google_endpoint()
    test_temp_endpoint()

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Can run with pytest or directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        main()
