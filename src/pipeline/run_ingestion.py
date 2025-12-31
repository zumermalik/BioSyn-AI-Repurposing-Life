import os
import subprocess
import sys

def install_node_deps():
    """Installs npm packages if strictly necessary (for Colab)"""
    print("📦 Checking Node.js dependencies...")
    if not os.path.exists("node_modules"):
        print("   >> Running npm install...")
        subprocess.run(["npm", "install"], check=True)
    else:
        print("   >> node_modules found. Skipping install.")

def run_ts_ingest():
    """Executes the TypeScript Ingestion Engine"""
    ts_script = "src/ingestion/fetch_pdb.ts"
    
    print(f"🚀 Launching TS Ingestion Engine: {ts_script}")
    # We use npx ts-node to run without global install
    try:
        subprocess.run(["npx", "ts-node", ts_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_node_deps()
    run_ts_ingest()