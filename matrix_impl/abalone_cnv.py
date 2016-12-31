import sys
import os

def convert(fname):
    out = []
    with open(fname) as f:
        content = f.readlines()
        print(content[0:3])
        for elem in content:
            out.append(elem[2:])
    with open(os.path.splitext(fname)[0] + "_cnv.txt", 'w+') as f:
        f.write(''.join(out))

fname = sys.argv[1]
print(fname)
convert(fname)
