import os

import httpx


async def google_search(query: str) -> str:
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
        )
        results = response.json()
        results = "\n".join(
            [result["link"] for result in results["items"][:3]]
        )  # just want the first three, only the links
        return results
