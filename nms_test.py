import os 
from utills import non_max_supression


class TestNonMaxSuppression():
    def __init__(self):
        # test cases we want to run
        self.t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c1_boxes = [[1, 1, 0.5, 0.45, 0.4, 0.5], [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

        self.t2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

        self.t3_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c3_boxes = [[1, 1, 0.5, 0.5, 0.2, 0.4], [2, 0.8, 0.25, 0.35, 0.3, 0.1]]

        self.t4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

    def runTests(self):
        print("========start=========")
        # self.r1  =  non_max_supression(self.t1_boxes ,   7 / 20  , 0.2 )
        # self.r2  =  non_max_supression(self.t2_boxes ,   7 / 20  , 0.2 )
        # self.r3  =  non_max_supression(self.t3_boxes ,   7 / 20  , 0.2 )
        self.r4  =  non_max_supression(self.t4_boxes ,   9 / 20  , 0.2 )
        print(self.r4)



        # assert(self.r1 == self.c1_boxes)    
        # assert(self.r2 == self.c2_boxes)    
        # assert(self.r3 == self.c3_boxes)    
        assert(self.r4 == self.c4_boxes)    


        print("\n\n***===== Suceess===*** : All 4 tests passed.\n\n")

if __name__ == "__main__":
    tests =  TestNonMaxSuppression()
    tests.runTests()