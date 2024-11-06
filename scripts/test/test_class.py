class A:

    def __call__(self):
        print('a')

class B(A):

    def __call__(self, repair_action):
        print(repair_action)


if __name__ == '__main__':

    a = A()

    a()

    b = B()

    b('test')