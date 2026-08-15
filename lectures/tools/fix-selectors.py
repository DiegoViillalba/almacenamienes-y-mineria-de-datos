with open('unam-theme.scss', 'r') as f:
    content = f.read()

content = content.replace(
    '#title-slide .quarto-title-affiliation:after,\n#title-slide .date:after,\n#title-slide .author:after {',
    '#title-slide .quarto-title-affiliation:after,\n#title-slide .date:after,\n#title-slide .author:after,\n#title-slide .quarto-title-author-name:after {'
)

with open('unam-theme.scss', 'w') as f:
    f.write(content)

print("Fixed")
