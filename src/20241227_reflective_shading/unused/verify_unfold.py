import numpy as np
from sh_utils import unfold_sh_coeff

def main():
    data = np.array([0, -11, 10, 11, -22, -21, 20, 21, 22])
    data = np.concatenate([data[None], data[None], data[None]], axis=0)
    data = unfold_sh_coeff(data)
    print(data[0])
    exit()

if __name__ == "__main__":
    main()