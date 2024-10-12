from LineNotify import LineNotify

line = LineNotify()
#line.send("Hello, World!")
a = line.send_image("Ball", "output.jpg")
print(a)