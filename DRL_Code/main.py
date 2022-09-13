# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    natuals = itertools.count(2)  # infinite iterator
    for n in natuals:
        print(n)
        print(111)
        if n >= 100:
            break

    tensor_0 = torch.arange(3, 12).view(3, 3)
    print(tensor_0)

    # Index tensor must have the same number of dimensions as input tensor
    index = torch.tensor([[2, 1, 0]])
    tensor_1 = tensor_0.gather(1, index)
    print(tensor_1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
