import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent"))

# Constructing the real Anthropic client needs a key to exist; tests never call the API.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
