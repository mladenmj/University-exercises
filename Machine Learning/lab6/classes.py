class Hello:
	GREET = "Hi"
	
	def __init__(self, initial_name):
		self.name = initial_name
	
	def change_name(self, new_name):
		self.name = new_name
	
	def say_hello(self):
		print(self.GREET, self.name, "!\n")

h = Hello("Karl")
h.change_name("Marko")
h.say_hello()

##########################################################
import math

class Point(object):
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y
	
	def __add__(self, p):
		newpoint = Point()
		newpoint.x = self.x + p.x
		newpoint.y = self.y + p.y
		return newpoint
	
	def __repr__(self):
		return "Point(" + str(self.x) + "," + str(self.y) + ")"
	
	def move_by(self, dx, dy):
		self.x += dx
		self.y += dy
	
	def distance(self, p):
		d2 = (self.x - p.x) ** 2 + (self.y - p.y) ** 2
		return math.sqrt(d2)

p = Point()
p.move_by(3, 4)
q = Point(6, 7)

s = p + q
print(s)
