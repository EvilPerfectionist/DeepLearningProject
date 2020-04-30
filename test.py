aa = [1, 2, 3, 4]
aa.append(5)
aa.pop(0)
print(aa[0])
print(len(aa))

bb = sum(i > 2 for i in aa)
print(bb)
