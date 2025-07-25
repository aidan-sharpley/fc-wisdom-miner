#!/usr/bin/env python3
"""
Model setup script for optimized M1 MacBook Air performance.
Installs and verifies the recommended LLM models for the forum analyzer.
"""

import subprocess
import sys
import time
import requests
from typing import List, Dict

# Recommended models for M1 8GB setup
RECOMMENDED_MODELS = {
    'qwen2.5:0.5b': {
        'purpose': 'Ultra-fast analytics and structured data',
        'memory': '~0.8GB',
        'speed': 'Very Fast',
        'priority': 'High'
    },
    'qwen2.5:1.5b': {
        'purpose': 'Fast narrative generation',
        'memory': '~1.2GB', 
        'speed': 'Fast',
        'priority': 'High'
    },
    'llama3.2:3b': {
        'purpose': 'High-quality narratives (optional)',
        'memory': '~2.4GB',
        'speed': 'Medium',
        'priority': 'Medium'
    },
    'deepseek-r1:1.5b': {
        'purpose': 'Current chat model',
        'memory': '~1.2GB',
        'speed': 'Fast', 
        'priority': 'High'
    },
    'nomic-embed-text:v1.5': {
        'purpose': 'Text embeddings',
        'memory': '~0.5GB',
        'speed': 'Fast',
        'priority': 'Critical'
    }
}

def run_command(cmd: List[str]) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

def get_installed_models() -> List[str]:
    """Get list of currently installed Ollama models."""
    success, output = run_command(['ollama', 'list'])
    if not success:
        return []
    
    models = []
    for line in output.split('\n')[1:]:  # Skip header
        if line.strip():
            model_name = line.split()[0]
            if ':' in model_name:
                models.append(model_name)
    return models

def install_model(model_name: str) -> bool:
    """Install a specific model."""
    print(f"Installing {model_name}...")
    success, output = run_command(['ollama', 'pull', model_name])
    if success:
        print(f"✅ {model_name} installed successfully")
        return True
    else:
        print(f"❌ Failed to install {model_name}: {output}")
        return False

def test_model(model_name: str) -> bool:
    """Test if a model responds correctly."""
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        response = requests.post('http://localhost:11434/api/chat', json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return bool(result.get('message', {}).get('content', '').strip())
        return False
    except:
        return False

def estimate_memory_usage(models: List[str]) -> float:
    """Estimate total memory usage for given models."""
    memory_map = {
        'qwen2.5:0.5b': 0.8,
        'qwen2.5:1.5b': 1.2, 
        'qwen2.5:3b': 2.4,
        'llama3.2:3b': 2.4,
        'deepseek-r1:1.5b': 1.2,
        'nomic-embed-text:v1.5': 0.5
    }
    
    total = 0
    for model in models:
        total += memory_map.get(model, 1.0)  # Default 1GB if unknown
    return total

def main():
    print("🚀 Forum Analyzer Model Setup for M1 MacBook Air (8GB RAM)")
    print("=" * 60)
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first:")
        print("   brew install ollama")
        print("   ollama serve")
        sys.exit(1)
    
    print("✅ Ollama is running")
    
    # Get currently installed models
    installed_models = get_installed_models()
    print(f"\n📦 Currently installed models: {len(installed_models)}")
    for model in installed_models:
        status = "✅" if model in RECOMMENDED_MODELS else "ℹ️"
        print(f"  {status} {model}")
    
    # Determine which models to install
    missing_models = []
    optional_models = []
    
    for model, info in RECOMMENDED_MODELS.items():
        if model not in installed_models:
            if info['priority'] in ['Critical', 'High']:
                missing_models.append(model)
            else:
                optional_models.append(model)
    
    if not missing_models and not optional_models:
        print("\n✅ All recommended models are installed!")
    else:
        print(f"\n📋 Installation Plan:")
        
        # Show required models
        if missing_models:
            print("  Required models:")
            for model in missing_models:
                info = RECOMMENDED_MODELS[model]
                print(f"    • {model} - {info['purpose']} ({info['memory']})")
        
        # Show optional models
        if optional_models:
            print("  Optional models:")
            for model in optional_models:
                info = RECOMMENDED_MODELS[model]
                print(f"    • {model} - {info['purpose']} ({info['memory']})")
        
        # Memory estimation
        all_to_install = missing_models + optional_models
        estimated_memory = estimate_memory_usage(all_to_install + installed_models)
        print(f"\n💾 Estimated total memory usage: ~{estimated_memory:.1f}GB")
        
        if estimated_memory > 6:
            print("⚠️  Warning: This may exceed 8GB RAM. Consider installing only required models.")
        
        # Ask for confirmation
        response = input("\n🤔 Proceed with installation? (y/N): ").lower().strip()
        if response != 'y':
            print("Installation cancelled.")
            sys.exit(0)
        
        # Install models
        print("\n🔄 Installing models...")
        successful_installs = []
        failed_installs = []
        
        for model in missing_models:
            if install_model(model):
                successful_installs.append(model)
            else:
                failed_installs.append(model)
        
        # Ask about optional models
        if optional_models and not failed_installs:
            response = input("\n🤔 Install optional models too? (y/N): ").lower().strip()
            if response == 'y':
                for model in optional_models:
                    if install_model(model):
                        successful_installs.append(model)
                    else:
                        failed_installs.append(model)
    
    # Test installed models
    print("\n🧪 Testing model responses...")
    all_models = get_installed_models()
    working_models = []
    broken_models = []
    
    for model in all_models:
        if model in RECOMMENDED_MODELS:
            print(f"Testing {model}...", end=" ")
            if test_model(model):
                print("✅")
                working_models.append(model)
            else:
                print("❌")
                broken_models.append(model)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    print(f"  ✅ Working models: {len(working_models)}")
    if broken_models:
        print(f"  ❌ Broken models: {len(broken_models)}")
    
    estimated_memory = estimate_memory_usage(working_models)
    print(f"  💾 Memory usage: ~{estimated_memory:.1f}GB")
    
    if len(working_models) >= 3:
        print("\n🎉 Setup complete! Your system is optimized for forum analysis.")
        print("\nRecommended configuration:")
        print("  • Analytics: qwen2.5:0.5b (ultra-fast)")
        print("  • Narratives: qwen2.5:1.5b (fast)")
        print("  • Fallback: qwen2.5:0.5b (reliable)")
    else:
        print("\n⚠️  Setup incomplete. Install missing models to ensure optimal performance.")
    
    print(f"\n🚀 Ready to run: uv run python app.py")

if __name__ == "__main__":
    main()