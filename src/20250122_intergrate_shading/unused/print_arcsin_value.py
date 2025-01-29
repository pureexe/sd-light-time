
import numpy as np

def inspect_arcsin_value(z):
    print(f"z: {z:.2f}")
    print("arcsin(z): ", np.arcsin(z) / np.pi , "π")
    print("=====================================")

def main():
    inspect_arcsin_value(1)
    inspect_arcsin_value(0)
    inspect_arcsin_value(-1)

if __name__ == "__main__":
    main()