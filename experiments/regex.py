import re

f = open("censorship-of-lgbt-issues.csv", "r")
text = f.readlines()
found = []
for l in text :
    pattern = re.compile('[A-Z]{3},2023,.*')
    p = pattern.search(l)
    if p != None:
        s = l[p.start() : p.end()]
        s = re.sub(',2023,', ',', s)
        found.append(s +  "\n")

print(found)
fw = open("censorship-of-lgbt-issues-2023.csv", "w")

fw.writelines(found)
fw.close()
f.close()


print(found)