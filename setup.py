import subprocess
import requests
from pathlib import Path
import sys

# Step 1: Install dependencies
print("📦 Installing dependencies...")

venv_path = Path(".venv")

if not (venv_path.exists() and venv_path.is_dir()):
    print("⚠️  No virtual environment found (.venv directory missing).")
    create = input("Would you like to create one? [y/N]: ").strip().lower()
    if create == "y":
        import subprocess

        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("✅ Virtual environment created at .venv/")
    else:
        print("❌ Setup aborted. Please create a virtual environment first.")
        exit(1)
else:
    print("✅ Using existing virtual environment.")

if venv_path.exists() and sys.platform.startswith("win"):
    cmd = r"venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    subprocess.run(["powershell", "-Command", cmd])

# Step 2: Get Hugging Face API key
api_key = input("\n🔑 Enter your Hugging Face API key: ").strip()

# Step 3: Verify API key
print("🔍 Checking if your Hugging Face API key works...")
try:
    response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if response.status_code == 200:
        user = response.json()
        print(f"✅ Token valid! Logged in as: {user.get('name', 'Unknown')}")
    else:
        print(f"❌ Invalid API key or request failed (HTTP {response.status_code})")
        exit(1)
except requests.RequestException as e:
    print(f"❌ Network or API error: {e}")
    exit(1)

# Step 4: Replace or store API key
env_path = Path("pipeline/.env")

print("💾 Saving API key to pipeline/.env")
env_path.write_text(f"HUGGINGFACE_API_KEY={api_key}\n")

# Step 5: Run audio_detect.py inside pipeline
print("\n🚀 Starting audio detection pipeline...")
subprocess.run(["python", "-m", "pipeline.audio_detect"], cwd=".", check=True)
