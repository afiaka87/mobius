import asyncio
import sys
sys.path.insert(0, '/Users/claym/projects/mobius-refactored')

async def test_celeste():
    import services
    try:
        print("Testing Celeste API at http://192.168.1.216:8000...")
        result = await services.generate_celeste_image(
            prompt="a beautiful sunset over mountains",
            negative_prompt="blurry, low quality",
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42,
            width=512,
            height=512
        )
        print(f"Success! Image saved to: {result}")
        return result
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_celeste())
    if result:
        print(f"\nTest completed successfully. Image path: {result}")
    else:
        print("\nTest failed.")
