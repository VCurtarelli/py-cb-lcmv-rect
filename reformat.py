with open('main.py', 'r', encoding='utf8') as f:
    text = f.read()
    f.close()

lines = text.split('\n')

l_idx = 0
while l_idx < len(lines) - 2:
    sps = 0
    if lines[l_idx] == lines[l_idx+2] and lines[l_idx].endswith('---'):
        line1 = "'''"
        line3 = "'''"
        line2 = lines[l_idx+1]
        line2 = '    ' + line2[line2.find('#')+2:]
        space = lines[l_idx][:lines[l_idx].find('#')]
        lines[l_idx+0] = space + line1
        lines[l_idx+1] = space + line2
        lines[l_idx+2] = space + line3

        l_idx += 2
    l_idx += 1

text = '\n'.join(lines)
# print(len(text))
# text = text.replace('\t', '    ')
# print(len(text))
with open('mainv2.py', 'w', encoding='utf8') as f:
    f.write(text)
    f.close()