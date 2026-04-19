#!/usr/bin/env python3
"""
Application entry point for the ISLES stroke lesion segmentation system.
"""
import os

from app import app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true'
    )
