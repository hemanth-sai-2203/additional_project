import sys
import os

def check_imports():
    packages = [
        "numpy", "scipy", "pandas", "sklearn", "matplotlib", 
        "torch", "stable_baselines3", "gymnasium"
    ]
    
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    
    missing = []
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {pkg:<20} | Version: {version}")
        except ImportError:
            print(f"[FAIL] {pkg:<20} | NOT FOUND")
            missing.append(pkg)

    print("-" * 30)
    try:
        import carla
        print("[OK] carla                | SUCCESS")
    except ImportError:
        print("[!] carla                | NOT FOUND (Reminder: Install the .whl file)")

    if not missing:
        print("\nAll core dependencies are installed successfully!")
    else:
        print(f"\nMissing dependencies: {', '.join(missing)}")

if __name__ == "__main__":
    check_imports()

