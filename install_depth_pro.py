"""
Installation script for Depth Pro
"""

import os
import subprocess
import sys
import requests
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def download_file(url, destination):
    """Download a file from URL"""
    print(f"\n🔄 Downloading {os.path.basename(destination)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def check_depth_pro_installation():
    """Check if Depth Pro is already installed"""
    try:
        import depth_pro
        print("✓ Depth Pro is already installed!")
        return True
    except ImportError:
        print("⚠ Depth Pro is not installed")
        return False

def install_depth_pro():
    """Install Depth Pro step by step"""
    print("=" * 60)
    print("DEPTH PRO INSTALLATION")
    print("=" * 60)
    
    # Check if already installed
    if check_depth_pro_installation():
        print("Depth Pro is already available. No installation needed.")
        return True
    
    # Create a temporary directory for installation
    temp_dir = Path("temp_depth_pro")
    original_dir = Path.cwd()
    
    try:
        # Step 1: Clone the repository
        if not run_command(
            "git clone https://github.com/apple/ml-depth-pro.git temp_depth_pro",
            "Cloning Depth Pro repository"
        ):
            return False
        
        # Step 2: Change to the cloned directory
        os.chdir(temp_dir)
        
        # Step 3: Install Depth Pro
        if not run_command(
            f"{sys.executable} -m pip install -e .",
            "Installing Depth Pro package"
        ):
            os.chdir(original_dir)
            return False
        
        # Step 4: Create checkpoints directory
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Step 5: Download the pretrained model
        model_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
        model_path = checkpoints_dir / "depth_pro.pt"
        
        if not download_file(model_url, str(model_path)):
            print("⚠ Model download failed, but Depth Pro package is installed")
            print("You can manually download the model later")
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Step 6: Verify installation
        print("\n🔄 Verifying installation...")
        if check_depth_pro_installation():
            print("\n🎉 Depth Pro installation completed successfully!")
            return True
        else:
            print("\n❌ Installation verification failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        os.chdir(original_dir)
        return False
    finally:
        # Cleanup
        if temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print("🧹 Cleaned up temporary files")
            except Exception as e:
                print(f"⚠ Could not clean up temporary files: {e}")

def install_dependencies():
    """Install required dependencies"""
    print("\n🔄 Installing dependencies...")
    
    dependencies = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0"
    ]
    
    for dep in dependencies:
        if not run_command(
            f"{sys.executable} -m pip install '{dep}'",
            f"Installing {dep}"
        ):
            print(f"⚠ Failed to install {dep}, continuing...")
    
    print("✓ Dependencies installation completed")

def test_depth_pro():
    """Test Depth Pro installation"""
    print("\n" + "=" * 60)
    print("TESTING DEPTH PRO")
    print("=" * 60)
    
    try:
        import depth_pro
        import torch
        import numpy as np
        from PIL import Image
        
        print("✓ Depth Pro imports successful")
        
        # Create a test image
        test_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        # Load model
        model, transform = depth_pro.create_model_and_transforms()
        model.eval()
        
        print("✓ Model loading successful")
        
        # Test inference
        image_tensor = transform(test_image).unsqueeze(0)
        with torch.no_grad():
            prediction = model.infer(image_tensor)
            depth_map = prediction["depth"].cpu().numpy().squeeze()
        
        print(f"✓ Depth estimation successful: {depth_map.shape}")
        print(f"✓ Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Depth Pro test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("DEPTH PRO INSTALLATION SCRIPT")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Install dependencies first
    install_dependencies()
    
    # Install Depth Pro
    if install_depth_pro():
        # Test the installation
        if test_depth_pro():
            print("\n🎉 SUCCESS! Depth Pro is now ready to use.")
            print("\nYou can now restart your Flask app and it will use Depth Pro for better depth estimation.")
        else:
            print("\n⚠ Installation completed but testing failed.")
            print("The fallback depth estimation will still work.")
    else:
        print("\n❌ Installation failed.")
        print("The system will continue to use the improved fallback depth estimation.")
    
    return True

if __name__ == "__main__":
    main()
