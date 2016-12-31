import sys
import os

def addlinenum(fname):
    out = []
    linenum = 1
    with open(fname) as f:
        content = f.readlines()
        for elem in content:
            line = str(linenum) + "," + elem
            linenum += 1
            out.append(line)
    with open(os.path.splitext(fname)[0] + "_with_linenum.txt", 'w+') as f:
        f.write(''.join(out))

fname = sys.argv[1]
addlinenum(fname)
