import os
from telegram import Bot
import asyncio
import threading
from io import BytesIO
from dotenv import load_dotenv

# should be placed in the root directory under .env
load_dotenv()


class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Flag to check if the notifier is properly configured
        self.is_configured = bool(self.token and self.chat_id)

        if not self.is_configured:
            print("Warning: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set. Telegram notifications will be disabled.")
            # Use dummy values to avoid errors
            self.token = "dummy_token"
            self.chat_id = "dummy_chat_id"

        self.bot = Bot(token=self.token)
        self.loop = None
        self.thread = None
        self.start()

    def start(self):
        """Initialize the event loop and start the background thread"""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, args=(self.loop,), daemon=True)
        self.thread.start()

    def _run_event_loop(self, loop):
        """Run the event loop in the background thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def stop(self):
        """Stop the event loop and join the thread"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()

    async def _send_message_async(self, message: str):
        """Asynchronously sends a text message to the specified Telegram chat."""
        # Skip if not properly configured
        if not self.is_configured:
            return

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
        except Exception as e:
            print(f"Failed to send message via Telegram: {e}")

    async def _send_image_async(self, image_bytes: bytes, caption: str = ""):
        """Asynchronously sends an image to the specified Telegram chat."""
        # Skip if not properly configured
        if not self.is_configured:
            return

        try:
            image_buffer = BytesIO(image_bytes)
            await self.bot.send_photo(chat_id=self.chat_id, photo=image_buffer, caption=caption)
        except Exception as e:
            print(f"Failed to send image via Telegram: {e}")

    def send_message(self, message: str):
        """Synchronously sends a text message by submitting it to the background event loop."""
        # Skip if not properly configured
        if not self.is_configured:
            return

        asyncio.run_coroutine_threadsafe(self._send_message_async(message), self.loop)

    def send_image(self, image_bytes: bytes, caption: str = ""):
        """Synchronously sends an image by submitting it to the background event loop."""
        # Skip if not properly configured
        if not self.is_configured:
            return

        asyncio.run_coroutine_threadsafe(self._send_image_async(image_bytes, caption), self.loop)


notifier = TelegramNotifier()
