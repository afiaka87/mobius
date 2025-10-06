# views.py

import asyncio
import logging
from typing import Any

import discord

import services
import utils
from autocomplete import autocomplete_context

logger = logging.getLogger(__name__)


# --- Action Buttons ---

class MediaActionButton(discord.ui.Button["PlaySelectView"]):
    """A button that performs an action (play/queue) on a media item."""

    def __init__(self, media: dict[str, Any], action: str, **kwargs: Any) -> None:
        self.media = media
        self.action = action
        super().__init__(**kwargs)

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            media_name = self.media.get("display_name") or self.media.get("name") or "Unknown"
            media_id = self.media["id"]

            if self.action == "play":
                await services.play_obscast_media(media_id)
                embed = utils.create_success_embed(
                    f"Now playing **{media_name}**.", title="▶️ Playback Started"
                )
                autocomplete_context.add_user_play(
                    interaction.user.id, media_id, media_name
                )
            elif self.action == "queue":
                result = await services.queue_obscast_media(media_id)
                pos = result.get("queue_position", "?")
                embed = utils.create_success_embed(
                    f"Added **{media_name}** to queue at position **#{pos}**.",
                    title="📋 Media Queued",
                )
            else:
                raise ValueError("Invalid action")

            await interaction.followup.send(embed=embed, ephemeral=True)

            # Disable buttons on the original message
            if self.view:
                for item in self.view.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = True
                await self.view.message.edit(view=self.view)
                self.view.stop()

        except services.ObscastAPIError as e:
            await interaction.followup.send(
                embed=utils.create_error_embed(f"Failed to perform action: {e}"),
                ephemeral=True,
            )


# --- Selection Views ---

class PlaySelectView(discord.ui.View):
    """A view with buttons (1-5) for quick selection from search results."""

    message: discord.Message

    def __init__(self, results: list[dict[str, Any]], query: str) -> None:
        super().__init__(timeout=180.0)
        self.query = query

        for i, media in enumerate(results[:5]):
            button = MediaActionButton(
                media=media,
                action="play",
                label=f"{i+1}",
                style=discord.ButtonStyle.secondary,
                row=0,
            )
            self.add_item(button)

    async def on_timeout(self) -> None:
        try:
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True
            await self.message.edit(view=self)
        except discord.NotFound:
            pass  # Message was likely deleted


class SearchResultsView(discord.ui.View):
    """A view that presents search results with play and queue buttons."""

    message: discord.Message

    def __init__(self, results: list[dict[str, Any]], query: str) -> None:
        super().__init__(timeout=300.0)
        self.results = results
        self.query = query
        self.page = 0
        self.per_page = 5
        self.update_view()

    @property
    def max_pages(self) -> int:
        return (len(self.results) - 1) // self.per_page

    def update_view(self) -> None:
        """Clears and re-adds all components for the current page."""
        self.clear_items()
        start_index = self.page * self.per_page
        end_index = start_index + self.per_page
        current_page_results = self.results[start_index:end_index]

        for i, media in enumerate(current_page_results):
            play_button = MediaActionButton(
                media, "play", label="▶️ Play", style=discord.ButtonStyle.success, row=i
            )
            queue_button = MediaActionButton(
                media, "queue", label="📋 Queue", style=discord.ButtonStyle.secondary, row=i
            )
            self.add_item(play_button)
            self.add_item(queue_button)

        # Navigation buttons (simplified for line length)
        row = self.per_page
        prev_disabled = self.page == 0
        next_disabled = self.page >= self.max_pages

        prev_button = discord.ui.Button(
            label="⬅️",
            style=discord.ButtonStyle.primary,
            row=row,
            disabled=prev_disabled
        )
        next_button = discord.ui.Button(
            label="➡️",
            style=discord.ButtonStyle.primary,
            row=row,
            disabled=next_disabled
        )
        page_label = f"Page {self.page + 1}/{self.max_pages + 1}"
        page_indicator = discord.ui.Button(
            label=page_label,
            style=discord.ButtonStyle.secondary,
            row=row,
            disabled=True
        )

        prev_button.callback = self.prev_page
        next_button.callback = self.next_page

        self.add_item(prev_button)
        self.add_item(page_indicator)
        self.add_item(next_button)

    def create_embed(self) -> discord.Embed:
        return utils.create_search_results_embed(self.results, self.query, self.page, self.per_page)

    async def prev_page(self, interaction: discord.Interaction) -> None:
        self.page -= 1
        self.update_view()
        await interaction.response.edit_message(embed=self.create_embed(), view=self)

    async def next_page(self, interaction: discord.Interaction) -> None:
        self.page += 1
        self.update_view()
        await interaction.response.edit_message(embed=self.create_embed(), view=self)

    async def on_timeout(self) -> None:
        try:
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True
            await self.message.edit(view=self)
        except discord.NotFound:
            pass


# --- Image Search Results View ---

class ImageSearchResultsView(discord.ui.View):
    """A view that displays image search results with thumbnails."""

    message: discord.Message

    def __init__(self, results: list[dict[str, Any]], query: str) -> None:
        super().__init__(timeout=300.0)
        self.results = results
        self.query = query
        self.current_index = 0
        self.update_view()

    def update_view(self) -> None:
        """Updates navigation buttons based on current index."""
        self.clear_items()

        # Add navigation buttons if we have multiple results
        if len(self.results) > 1:
            prev_button = discord.ui.Button(
                label="◀",
                style=discord.ButtonStyle.primary,
                disabled=self.current_index == 0
            )
            next_button = discord.ui.Button(
                label="▶",
                style=discord.ButtonStyle.primary,
                disabled=self.current_index >= len(self.results) - 1
            )
            position_label = discord.ui.Button(
                label=f"{self.current_index + 1}/{len(self.results)}",
                style=discord.ButtonStyle.secondary,
                disabled=True
            )

            prev_button.callback = self.prev_image
            next_button.callback = self.next_image

            self.add_item(prev_button)
            self.add_item(position_label)
            self.add_item(next_button)

    def create_embed(self) -> discord.Embed:
        """Creates an embed for the current image."""
        if not self.results:
            return utils.create_error_embed("No images found.")

        result = self.results[self.current_index]
        embed = discord.Embed(
            title=f"🔍 Image Search: {self.query}",
            color=discord.Color.blue()
        )

        # Add image URL as the main image
        image_url = result.get("url", "")
        if image_url:
            embed.set_image(url=image_url)

        # Add metadata fields
        if caption := result.get("caption"):
            # Truncate caption if too long
            if len(caption) > 200:
                caption = caption[:197] + "..."
            embed.add_field(name="Caption", value=caption, inline=False)

        if similarity := result.get("similarity"):
            embed.add_field(name="Similarity", value=f"{similarity:.2%}", inline=True)

        if nsfw := result.get("NSFW"):
            embed.add_field(name="Safety", value=nsfw, inline=True)

        # Add footer with position info
        embed.set_footer(text=f"Result {self.current_index + 1} of {len(self.results)}")

        return embed

    async def prev_image(self, interaction: discord.Interaction) -> None:
        """Navigate to previous image."""
        self.current_index = max(0, self.current_index - 1)
        self.update_view()
        await interaction.response.edit_message(embed=self.create_embed(), view=self)

    async def next_image(self, interaction: discord.Interaction) -> None:
        """Navigate to next image."""
        self.current_index = min(len(self.results) - 1, self.current_index + 1)
        self.update_view()
        await interaction.response.edit_message(embed=self.create_embed(), view=self)

    async def on_timeout(self) -> None:
        try:
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True
            await self.message.edit(view=self)
        except discord.NotFound:
            pass


# --- Now Playing View ---

class NowPlayingView(discord.ui.View):
    """The unified control center for the currently playing media."""

    message: discord.Message

    def __init__(self, current: dict[str, Any], queue: dict[str, Any]) -> None:
        super().__init__(timeout=None)  # Persistent view
        self.current = current
        self.queue = queue
        self.update_buttons()

    def update_buttons(self) -> None:
        """Updates the state of buttons based on playback status."""
        is_playing = self.current.get("is_playing", False)

        play_pause_button = discord.utils.find(lambda i: i.custom_id == "play_pause", self.children)
        if play_pause_button:
            play_pause_button.label = "Pause" if is_playing else "Play"
            play_pause_button.emoji = "⏸️" if is_playing else "▶️"
            play_pause_button.style = discord.ButtonStyle.primary if is_playing else discord.ButtonStyle.success

    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️", custom_id="play_pause", row=0)
    async def play_pause_button(self, interaction: discord.Interaction, __button: discord.ui.Button) -> None:
        """Toggles play/pause state."""
        await interaction.response.defer()
        action = "pause" if self.current.get("is_playing") else "resume"
        try:
            self.current = await services.control_obscast_playback(action)
            await self.refresh(interaction)
        except services.ObscastAPIError as e:
            await interaction.followup.send(embed=utils.create_error_embed(str(e)), ephemeral=True)

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary, emoji="⏭️", row=0)
    async def skip_button(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Skips to the next item in the queue."""
        await interaction.response.defer()
        try:
            await services.control_obscast_playback("skip")
            # Give the backend a moment to update
            await asyncio.sleep(1)
            await self.refresh(interaction)
        except services.ObscastAPIError as e:
            await interaction.followup.send(embed=utils.create_error_embed(str(e)), ephemeral=True)

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger, emoji="⏹️", row=0)
    async def stop_button(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Stops playback entirely."""
        await interaction.response.defer()
        try:
            await services.control_obscast_playback("stop")
            await interaction.followup.send(embed=utils.create_success_embed("Playback stopped."), ephemeral=True)
            # Disable the view after stopping
            for item in self.children:
                item.disabled = True
            await interaction.edit_original_response(view=self)
            self.stop()
        except services.ObscastAPIError as e:
            await interaction.followup.send(embed=utils.create_error_embed(str(e)), ephemeral=True)

    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.secondary, emoji="🔄", row=0)
    async def refresh_button(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        """Manually refreshes the Now Playing view."""
        await interaction.response.defer()
        await self.refresh(interaction)

    async def refresh(self, interaction: discord.Interaction) -> None:
        """Fetches fresh data and updates the message."""
        try:
            self.current = await services.get_obscast_current()
            self.queue = await services.get_obscast_queue()

            if not self.current.get("media_file"):
                await interaction.edit_original_response(
                    content="*Playback has ended.*", embed=None, view=None
                )
                self.stop()
                return

            self.update_buttons()
            embed = utils.create_now_playing_embed(self.current, self.queue)
            await interaction.edit_original_response(embed=embed, view=self)
        except services.ObscastAPIError as e:
            await interaction.edit_original_response(
                content=f"Could not refresh: {e}", embed=None, view=None
            )
            self.stop()

