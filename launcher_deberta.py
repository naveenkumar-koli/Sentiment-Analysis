import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Add project root and app dir to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from app.main_deberta import app

def open_browser(url: str, delay: float = 2.5):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    port = 8042
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 60)
    print("  Sales Sentiment Analysis Server - DeBERTa Transformer")
    print(f"  Access URL: {url}")
    print("  Status: Initializing fine-tuned transformer weights...")
    print("=" * 60)

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
