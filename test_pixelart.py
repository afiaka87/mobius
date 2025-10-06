#!/usr/bin/env python3
"""Test script for pixelart API integration."""

import asyncio
import logging
from services import generate_pixelart

logging.basicConfig(level=logging.INFO)


async def test_pixelart():
    """Test the pixelart generation function."""
    try:
        print("Testing pixelart generation...")
        print("API endpoint: http://100.70.95.57:8000/pixelart")

        # Test with a simple caption
        caption = "a cute pixel art cat"
        batch_size = 2
        seed = 123

        print(f"Caption: {caption}")
        print(f"Batch size: {batch_size}")
        print(f"Seed: {seed}")
        print("Generating...")

        images = await generate_pixelart(caption, batch_size, seed)

        print(f"✅ Success! Generated {len(images)} images")
        for i, img in enumerate(images):
            print(f"  - Image {i+1}: {len(img)} bytes")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_pixelart())