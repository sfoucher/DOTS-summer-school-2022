from typing import List

from ..data import Point, BoundingBox, PointProcessor, BboxProcessor


def bboxes_iou(bboxes_a: List[BoundingBox], bboxes_b: List[BoundingBox]) -> List[List[float]]:
    ''' Return Intersect-over-Union (IoU) of 2 sets of boxes.

    Both sets of boxes are expected to be in (x_min, y_min, x_max, y_max)
    format.

    Args:
        bboxes_a (list): list of BoundingBox objects (N)
        bboxes_b (list): list of BoundingBox objects (M)
    
    Returns:
        List[List[float]]:
            pairwise Intersect-over-Union (IoU) values stored in a list of
            shape NxM.
    '''

    assert all(isinstance(a, BoundingBox) for a in bboxes_a) is True, \
        'bboxes_a must contains BoundingBox objects only'
    
    assert all(isinstance(b, BoundingBox) for b in bboxes_b) is True, \
        'bboxes_b must contains BoundingBox objects only'

    nxm_shape_list = []
    for bbox_a in bboxes_a:

        m_shape_list = []
        for bbox_b in bboxes_b:
            intersect = BboxProcessor(bbox_a).intersect(bbox_b).area
            union = bbox_a.area + bbox_b.area - intersect

            m_shape_list.append(intersect/union)
        
        nxm_shape_list.append(m_shape_list)
    
    return nxm_shape_list

def points_dist(points_a: List[Point], points_b: List[Point]) -> List[List[float]]:
    ''' Return euclidean distances of 2 sets of points.

    Both sets of points are expected to be in (x, y) format.

    Args:
        points_a (list): list of Point objects (N)
        points_b (list): list of Point objects (M)
    
    Returns:
        List[List[float]]:
            pairwise euclidean distances values stored in a list of
            shape NxM.
    '''

    assert all(isinstance(a, Point) for a in points_a) is True, \
        'points_a must contains Point objects only'
    
    assert all(isinstance(b, Point) for b in points_b) is True, \
        'points_b must contains Point objects only'

    nxm_shape_list = []
    for point_a in points_a:

        m_shape_list = []
        for point_b in points_b:          
            dist = PointProcessor(point_a).dist(point_b)

            m_shape_list.append(dist)
        
        nxm_shape_list.append(m_shape_list)
    
    return nxm_shape_list