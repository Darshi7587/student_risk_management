"""Fix use_container_width warnings"""
import re

file_path = r'c:\Users\darsh\OneDrive\New folder\Desktop\omnitrics\app.py'

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace use_container_width=True with width="stretch"
content = content.replace('use_container_width=True', 'width="stretch"')

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed all use_container_width warnings!")
print("  Replaced with width='stretch'")
