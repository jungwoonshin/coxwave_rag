import os
import re
import pkg_resources

def find_imports(directory='.'):
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and virtual environments
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '.env', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Extract imports
    imports = set()
    import_pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)')
    
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                match = import_pattern.match(line)
                if match:
                    module_name = match.group(1).split('.')[0]
                    if module_name:
                        imports.add(module_name)
    
    return imports

def get_installed_packages():
    installed = {}
    for package in pkg_resources.working_set:
        installed[package.key] = package.version
    return installed

def is_standard_library(module_name):
    import sys
    
    if hasattr(sys, 'stdlib_module_names'):
        return module_name in sys.stdlib_module_names
    
    # Alternative approach for Python versions without stdlib_module_names
    try:
        path = __import__(module_name).__file__
        return path and ('site-packages' not in path and 'dist-packages' not in path)
    except (ImportError, AttributeError):
        return False

def main():
    imports = find_imports()
    installed_packages = get_installed_packages()
    
    # Filter out standard library modules
    third_party_imports = {module for module in imports if not is_standard_library(module)}
    
    # Only include packages with version numbers
    with open('requirements.txt', 'w') as f:
        for module in sorted(third_party_imports):
            module_lower = module.lower()
            
            # Direct match
            if module_lower in installed_packages:
                f.write(f"{module}=={installed_packages[module_lower]}\n")
                continue
                
            # Try with different casing
            for pkg in installed_packages:
                if pkg.lower() == module_lower:
                    f.write(f"{pkg}=={installed_packages[pkg]}\n")
                    break
    
    print(f"Generated requirements.txt with only versioned packages.")

if __name__ == "__main__":
    main()