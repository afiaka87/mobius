#!/bin/bash


# Create the main bot file
touch bot.py

# Create and populate the commands directory
mkdir commands
cd commands
touch __init__.py image_generation.py audio.py text.py utility.py video.py
cd ..

# Create and populate the services directory
mkdir services
cd services
touch __init__.py comfy_ui.py openai.py youtube.py weather.py
cd ..

# Create and populate the utils directory
mkdir utils
cd utils
touch __init__.py image_utils.py audio_utils.py video_utils.py
cd ..

# Create the root __init__.py file
touch __init__.py

echo "Mobius directory structure created successfully!"