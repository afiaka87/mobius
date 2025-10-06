# autocomplete.py

import logging
from collections import defaultdict, deque

import discord
from discord import app_commands

import services

logger: logging.Logger = logging.getLogger(__name__)


class AutocompleteContext:
    """Singleton class to track user patterns for better suggestions."""

    _instance: "AutocompleteContext | None" = None
    _initialized: bool

    def __new__(cls) -> "AutocompleteContext":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.user_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self.recent_plays: dict[int, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=10)
        )
        self._initialized = True
        logger.info("AutocompleteContext initialized.")

    def add_user_query(self, user_id: int, query: str) -> None:
        """Add a user query to their history."""
        if query and query.strip():
            self.user_history[user_id].appendleft(query.strip())

    def add_user_play(self, user_id: int, media_id: str, media_name: str) -> None:
        """Add a media item to a user's recent plays."""
        # Avoid duplicates
        self.recent_plays[user_id] = deque(
            [p for p in self.recent_plays[user_id] if p["id"] != media_id], maxlen=10
        )
        self.recent_plays[user_id].appendleft({"id": media_id, "name": media_name})


# Global instance
autocomplete_context = AutocompleteContext()


async def media_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """
    Provides intelligent autocomplete suggestions for media commands.

    - If the query is empty, it suggests recently played media.
    - If the query is active, it performs a search and boosts results
      based on user history and relevance.
    """
    suggestions: list[app_commands.Choice[str]] = []

    try:
        # Empty query: suggest recent plays
        if not current:
            recent_plays: deque[dict[str, str]] = autocomplete_context.recent_plays.get(interaction.user.id, deque())
            for play in recent_plays:
                name = f"🕐 {play['name']}"
                # Truncate to fit Discord limits
                if len(name) > 100:
                    name = name[:97] + "..."
                suggestions.append(app_commands.Choice(name=name, value=play["id"]))
            return suggestions[:25]

        # Active search
        results = await services.search_obscast_media(query=current, limit=25)

        # Format results as choices
        for media in results:
            name = media.get("display_name") or media.get("name", "Unknown")
            media_id = media.get("id", "unknown_id")
            media_type = media.get("type", "unknown")
            confidence = media.get("score", 0)

            # Add indicators for context
            emoji_map = {"movie": "🎬", "tv_show": "📺", "music": "🎵"}
            prefix = emoji_map.get(media_type, "📁")

            if confidence > 90:
                prefix = f"⭐ {prefix}"

            name = f"{prefix} {name}"

            # Truncate to fit Discord limits
            if len(name) > 100:
                name = name[:97] + "..."

            suggestions.append(app_commands.Choice(name=name, value=media_id))

        return suggestions[:25]  # Discord limit is 25

    except services.ObscastAPIError as e:
        logger.warning(f"Autocomplete failed due to API error: {e}")
        return [app_commands.Choice(name=f"⚠️ Error: {e}", value="error")]
    except Exception as e:
        logger.error(f"Unexpected error in media_autocomplete: {e}", exc_info=True)
        return [app_commands.Choice(name="⚠️ Autocomplete error", value="error")]

