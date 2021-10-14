import sys
import unittest
import tensorflow as torch

from utills import intersection_over_union

def test():
    # test cases we want to run
    # t1_box1 = torch.constant([0.8, 0.1, 0.2, 0.2])
    # t1_box2 = torch.constant([0.9, 0.2, 0.2, 0.2])
    # t1_correct_iou = 1 / 7

    t1_box1 = torch.constant([0.95, 0.6, 0.5, 0.2])
    t1_box2 = torch.constant([0.95, 0.7, 0.3, 0.2])
    t1_correct_iou = 3 / 13

    # t1_box1 = torch.constant([0.25, 0.15, 0.3, 0.1])
    # t1_box2 = torch.constant([0.25, 0.35, 0.3, 0.1])
    # t1_correct_iou = 0

    iou = intersection_over_union(t1_box1, t1_box2)
    print(iou)
    assert((torch.abs(iou - t1_correct_iou) < 0.001))


if __name__ == "__main__":
    print("Running Intersection Over Union Tests:")
    test()