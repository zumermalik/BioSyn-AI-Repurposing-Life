import sys
import yaml
import argparse
from pathlib import Path

# Add src to system path to ensure absolute imports work
sys.path.append(str(Path(__file__).parent.parent))

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="BioSyn AI Pipeline Runner")
    parser.add_argument('--mode', type=str, choices=['ingest', 'train', 'generate'], required=True, help="Pipeline mode")
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help="Path to config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print(f"🧬 BioSyn AI: Initializing Pipeline in [{args.mode.upper()}] mode...")
    print(f"🔧 Configuration loaded for project: {config['project_name']}")

    if args.mode == 'ingest':
        # TODO: Link the TS/Python ingestion module here
        print(">> Triggering data ingestion...")
    elif args.mode == 'train':
        # TODO: Link training loop
        print(f">> Starting training on device: {config['compute']['device']}")
    elif args.mode == 'generate':
        # TODO: Link inference
        print(">> Generating candidate molecules...")

if __name__ == "__main__":
    main()