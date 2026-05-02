#!/usr/bin/env python3
"""Download EasyOCR weights to a specified directory."""
import sys
import ssl
import easyocr

target_dir = sys.argv[1]
ssl._create_default_https_context = ssl._create_unverified_context
easyocr.Reader(["en"], gpu=False, model_storage_directory=target_dir, download_enabled=True, verbose=True)
print(f"EasyOCR weights saved to {target_dir}")
