# Create and populate the tests directory
mkdir tests
cd tests
touch __init__.py test_image_generation.py test_audio.py test_text.py test_utility.py test_video.py
cd ..

# Create a conftest.py file for shared pytest fixtures
touch conftest.py

echo "Mobius directory structure with tests created successfully!"