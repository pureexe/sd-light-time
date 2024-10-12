class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, my name is {self.name}!")

# Create an instance of the Person class
person = Person("Alice")

# Monkey patch to add a new method 'talk'
def talk(self):
    print(f"{self.name} is talking.")

# Add the 'talk' method to the Person class at runtime
Person.talk = talk

# Now you can call the newly added method on the instance
person.talk()
person.greet()