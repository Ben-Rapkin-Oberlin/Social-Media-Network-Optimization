

a=['a','b','c']
a=set(a)
b=['x','y','z']
b=set(b)
b.add('ww')

print(b-set('y'))
print(set([1]))
c=[['a','b'],['c','d'],['e','f']]
#print(set(c))
c=[set(x) for x in c]

