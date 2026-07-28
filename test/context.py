import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add src/sw directory to sys.path (Python pipeline lives under src/sw since
# the src/ split into src/sw (software) and src/hw (HLS/RTL))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'sw')))
