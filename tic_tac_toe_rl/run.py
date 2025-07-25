import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from main import main

if __name__ == "__main__":
    main() 