import os

print("This line is printed from test.py")

filename="/tmp/myfile.txt"

with open(filename, 'w') as out_file:
     out_file.write("This line was written by a python progam\n")
