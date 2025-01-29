
import numpy as np

def inspect_arctan2_value(x,y):
    print(f"x: {x:.2f}")
    print(f"y: {y:.2f}")
    print("arctan2(y,x): ", np.arctan2(y,x) / np.pi , "π")
    print("=====================================")

def main():
    inspect_arctan2_value(1,0)
    inspect_arctan2_value(0,1)
    inspect_arctan2_value(-1,0)
    inspect_arctan2_value(-np.sqrt(2),-np.sqrt(2))
    inspect_arctan2_value(0,-1)
    inspect_arctan2_value(np.sqrt(2),-np.sqrt(2))
    

if __name__ == "__main__":
    main()