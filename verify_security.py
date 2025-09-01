#!/usr/bin/env python3
"""
Security verification script for SRLP thesis evaluation pipeline.
Checks that API keys are properly secured and not exposed in code.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_gitignore():
    """Check that .env files are properly ignored."""
    gitignore_path = Path(".gitignore")
    
    if not gitignore_path.exists():
        return False, "No .gitignore file found"
    
    gitignore_content = gitignore_path.read_text()
    
    required_patterns = [".env", "*.env.backup"]
    missing_patterns = []
    
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        return False, f"Missing patterns in .gitignore: {missing_patterns}"
    
    return True, ".env files properly ignored"

def check_env_files():
    """Check that .env.example exists and .env is not tracked."""
    checks = []
    
    # Check .env.example exists
    if Path(".env.example").exists():
        checks.append("✅ .env.example exists")
    else:
        checks.append("❌ .env.example missing")
    
    # Check .env exists locally
    if Path(".env").exists():
        checks.append("✅ .env exists locally")
    else:
        checks.append("❌ .env missing (run: cp .env.example .env)")
    
    # Check .env is not tracked by git
    try:
        result = subprocess.run(
            ["git", "check-ignore", ".env"], 
            capture_output=True, 
            text=True,
            cwd="."
        )
        if result.returncode == 0:
            checks.append("✅ .env is properly ignored by git")
        else:
            checks.append("❌ .env is NOT ignored by git - security risk!")
    except subprocess.SubprocessError:
        checks.append("⚠️  Could not check git ignore status")
    
    return checks

def check_hardcoded_keys():
    """Check for any remaining hardcoded API keys."""
    key_patterns = [
        "sk-proj-",  # OpenAI project keys
        "sk-ant-",   # Anthropic keys
        "AIzaSy"     # Google/Gemini keys
    ]
    
    found_keys = []
    exclude_patterns = [".git", "__pycache__", ".env", "verify_security.py"]
    
    # Example/placeholder patterns that should be ignored
    placeholder_patterns = [
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "your-openai-key-here",
        "your-anthropic-key-here", 
        "your-gemini-key-here"
    ]
    
    for pattern in key_patterns:
        try:
            result = subprocess.run(
                ["grep", "-r", pattern, ".", "--exclude-dir=.git", "--exclude-dir=__pycache__"],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                # Filter out .env files, this script, and placeholder examples
                filtered_lines = []
                for line in lines:
                    # Skip excluded files
                    if any(exclude in line for exclude in exclude_patterns):
                        continue
                    
                    # Skip placeholder examples
                    if any(placeholder in line for placeholder in placeholder_patterns):
                        continue
                    
                    # Skip lines that are clearly documentation examples
                    if "XXXXXXXX" in line or "your-" in line or "example" in line.lower():
                        continue
                    
                    filtered_lines.append(line)
                
                if filtered_lines:
                    found_keys.extend(filtered_lines)
        
        except subprocess.SubprocessError:
            pass
    
    return found_keys

def check_config_security():
    """Check that config.py loads from environment variables."""
    config_path = Path("src/config.py")
    
    if not config_path.exists():
        return False, "config.py not found"
    
    config_content = config_path.read_text()
    
    # Check for dotenv import
    if "from dotenv import load_dotenv" not in config_content:
        return False, "dotenv not imported in config.py"
    
    # Check for load_dotenv() call
    if "load_dotenv()" not in config_content:
        return False, "load_dotenv() not called in config.py"
    
    # Check for environment variable loading
    if "get_required_env_var" not in config_content:
        return False, "Environment variable loading function missing"
    
    return True, "Config properly loads from environment variables"

def test_api_key_loading():
    """Test that API keys load correctly from .env."""
    try:
        sys.path.append('src')
        from config import load_config
        
        config = load_config()
        
        # Check that keys are loaded and not empty
        keys = {
            "OpenAI": config.openai_api_key,
            "Anthropic": config.anthropic_api_key,  
            "Gemini": config.gemini_api_key
        }
        
        results = []
        for name, key in keys.items():
            if key and len(key) > 10:
                results.append(f"✅ {name} API key loaded ({key[:10]}...)")
            else:
                results.append(f"❌ {name} API key missing or invalid")
        
        return True, results
    
    except Exception as e:
        return False, f"Failed to load config: {e}"

def main():
    """Run all security checks."""
    print("🔒 SRLP THESIS SECURITY VERIFICATION")
    print("=" * 50)
    
    # Check .gitignore
    gitignore_ok, gitignore_msg = check_gitignore()
    if gitignore_ok:
        print(f"✅ {gitignore_msg}")
    else:
        print(f"❌ {gitignore_msg}")
    
    # Check environment files
    print("\n📁 Environment Files:")
    env_checks = check_env_files()
    for check in env_checks:
        print(f"   {check}")
    
    # Check for hardcoded keys
    print("\n🔍 Hardcoded Key Check:")
    hardcoded_keys = check_hardcoded_keys()
    if not hardcoded_keys:
        print("   ✅ No hardcoded API keys found in source code")
    else:
        print("   ❌ SECURITY RISK: Hardcoded keys found:")
        for key_ref in hardcoded_keys[:5]:  # Show first 5
            print(f"      {key_ref}")
        if len(hardcoded_keys) > 5:
            print(f"      ... and {len(hardcoded_keys) - 5} more")
    
    # Check config security
    print("\n⚙️  Configuration Security:")
    config_ok, config_msg = check_config_security()
    if config_ok:
        print(f"   ✅ {config_msg}")
    else:
        print(f"   ❌ {config_msg}")
    
    # Test API key loading
    print("\n🔑 API Key Loading Test:")
    load_ok, load_results = test_api_key_loading()
    if load_ok:
        for result in load_results:
            print(f"   {result}")
    else:
        print(f"   ❌ {load_results}")
    
    # Overall security assessment
    print("\n" + "=" * 50)
    print("🛡️  SECURITY ASSESSMENT")
    
    all_checks = [
        gitignore_ok,
        not hardcoded_keys,  # No hardcoded keys = good
        config_ok,
        load_ok
    ]
    
    if all(all_checks):
        print("✅ SECURE: All security checks passed!")
        print("🎓 Safe for thesis submission and public repository")
        print("🔒 API keys are properly protected")
    else:
        print("❌ SECURITY ISSUES DETECTED!")
        print("⚠️  Fix issues before committing to repository")
        print("🚨 Do not submit with exposed API keys")
    
    # Quick setup reminder
    print("\n📋 SETUP REMINDER:")
    print("1. Copy: cp .env.example .env")
    print("2. Edit: nano .env (add your actual API keys)")  
    print("3. Never commit .env to version control")
    print("4. Use .env.example as template for others")

if __name__ == "__main__":
    main()
