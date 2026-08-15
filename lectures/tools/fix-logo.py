import re

with open('unam-theme.scss', 'r') as f:
    lines = f.readlines()

out_lines = []
data_url = ""

# First pass to find data_url
for line in lines:
    m = re.search(r"(url\('data:image/png;base64,[^']+'\))", line)
    if m:
        data_url = m.group(1)
        break

if not data_url:
    print("Not found")
    exit(1)

# Second pass to rewrite
i = 0
while i < len(lines):
    line = lines[i]
    if "/* Logo grande insertado antes del título en la portada */" in line:
        out_lines.append("/* Logo redimensionado insertado después del texto en la portada */\n")
        out_lines.append("#title-slide .quarto-title-affiliation:after,\n")
        out_lines.append("#title-slide .date:after,\n")
        out_lines.append("#title-slide .author:after {\n")
        out_lines.append('  content: "";\n')
        out_lines.append('  display: block;\n')
        out_lines.append('  width: 300px;\n')
        out_lines.append('  height: 120px;\n')
        out_lines.append('  margin: 40px auto 0 auto;\n')
        out_lines.append(f'  background-image: {data_url};\n')
        out_lines.append('  background-size: contain;\n')
        out_lines.append('  background-repeat: no-repeat;\n')
        out_lines.append('  background-position: center;\n')
        out_lines.append('}\n')
        
        # Skip original lines until '}'
        while i < len(lines) and not lines[i].strip() == "}":
            i += 1
        i += 1 # skip '}'
        continue
    
    out_lines.append(line)
    i += 1

with open('unam-theme.scss', 'w') as f:
    f.writelines(out_lines)

print("Success")
