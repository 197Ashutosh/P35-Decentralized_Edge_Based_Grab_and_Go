import os, re
for f in os.listdir('.'):
    if f.endswith('.py') and f != 'strip_comments.py':
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        content = re.sub(r'(?m)^\s*#.*$', '', content)
        content = re.sub(r'(?m)\s+#.*$', '', content)
        content = re.sub(r'\n\s*\n', '\n', content)
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
