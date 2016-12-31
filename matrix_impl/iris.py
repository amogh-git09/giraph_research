import sys
import os

def convert(fname):
    out = []
    with open(fname) as f:
        content = f.readlines()
        print(content[0:3])
        for elem in content:
            if elem == "\n":
                print("Skipping")
                continue
            c = 0
            tokens = elem.split(",")
            if tokens[-1] == "Iris-setosa\n":
                c = 0
            elif tokens[-1] == "Iris-versicolor\n":
                c = 1
            elif tokens[-1] == "Iris-virginica\n":
                c = 2

            line = ",".join(tokens[:-1]) + "," + str(c) + "\n"
            out.append(line)
        print(out[0:3])
    with open(os.path.splitext(fname)[0] + "_cnv.txt", 'w+') as f:
        f.write(''.join(out))

fname = sys.argv[1]
print(fname)
convert(fname)
