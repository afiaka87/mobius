import os
import subprocess
from pathlib import Path

import httpx


async def get_top_youtube_result_httpx(search_query, api_key):
    """
    Calls the YouTube Search API and fetches the top search result for a given query.

    Parameters:
    search_query (str): The search query string.
    api_key (str): Your YouTube Data API key.

    Returns:
    dict: Information about the top search result, or an error message if the call fails.
    """

    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "maxResults": 1,
        "key": api_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            if not items:
                return {"error": "No results found"}
            top_result = items[0]
            return {
                "videoId": top_result["id"]["videoId"],
                "title": top_result["snippet"]["title"],
                "description": top_result["snippet"]["description"],
                "channelTitle": top_result["snippet"]["channelTitle"],
            }
        else:
            return {
                "error": "Failed to fetch results, status code: {}".format(
                    response.status_code
                )
            }


async def download_youtube_video_as_audio(video_url: str):
    """Download a youtube video as an audio file.

    Args:
        video_url (str): The URL of the video to download. Must be a valid youtube URL. Example: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    Raises:
        Exception: A temporary file is created and subsequently uploaded. If the file is not created, an exception is raised.

    Returns:
        Path: The path to the downloaded audio file.
    """

    # scratch_filename = tempfile.NamedTemporaryFile(suffix=".opus", delete=False).name
    # the file is getting deleted before we can use it when using tempfile.NamedTemporaryFile
    # so we will use a /var/tmp/ file instead
    scratch_filename = "/var/tmp/yt_audio.opus"

    # Download the video as an audio file
    yt_dlp_command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "opus",
        "--output",
        scratch_filename,
        video_url,
    ]

    # Run the command
    subprocess.run(yt_dlp_command)

    # Check if the file exists
    if not os.path.exists(scratch_filename):
        raise Exception(f"Failed to download the video as an audio file: {video_url}")

    # Return the filename
    return Path(scratch_filename)
