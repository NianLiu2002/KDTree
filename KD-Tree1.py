from typing import List
from collections import namedtuple
import time


class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """
    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None
        self._n = 0
        self.nearestPoint=None
        self.nearestDistance=99999

    #https://blog.csdn.net/weixin_39011425/article/details/126105616
    def insert(self, p: List[Point])->Node:
        """insert a list of points"""
        def _build_KD_Tree(p: List[Point],depth:int=0)->Node:
            if not p:
                return None

            # Select axis based on depth so that axis cycles through all valid values
            axis = depth % 2

            # Sort point list by axis and choose median as pivot element
            p.sort(key=lambda x:x[axis])
            median = len(p) >>1
            median=get_left_position(p,axis,p[median][axis])

            # Create node and construct subtrees
            left_child = _build_KD_Tree(p[:median], depth + 1)
            right_child = _build_KD_Tree(p[median + 1:], depth + 1)
            self._n+=1
            return Node(
                location=p[median],
                left=left_child,
                right=right_child
            )
        self._root=_build_KD_Tree(p)

    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        result=[]
        def _dfs_range(node:Node,depth:int=0):
            if node is None:
                return
            if rectangle.is_contains(node.location):
                result.append(node.location)
            axis=depth%2
            if node.location[axis]>rectangle.lower[axis]:
                _dfs_range(node.left,depth+1)
            if node.location[axis]<=rectangle.upper[axis]:
                _dfs_range(node.right,depth+1)
        _dfs_range(self._root)
        return result

    
    def nearest_neighbor_query(self,query_point:Point)->Point:
        distance=lambda x,y:((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
        def _nearest_neighbor_query(node:Node,depth:int=0):
            
            if node.left==None and node.right==None:
                if distance(node.location,query_point)<self.nearestDistance:
                    self.nearestPoint=node.location
                    self.nearestDistance=distance(node.location,query_point)
                return #叶子节点返回
            axis=depth%2
            if query_point[axis]<node.location[axis]:#小的话往左走
                _nearest_neighbor_query(node.left,depth+1)
                if distance(node.location, query_point) < self.nearestDistance:
                    self.nearestPoint = node.location
                    self.nearestDistance = distance(node.location, query_point)
                if self.nearestDistance>abs(query_point[axis]-node.location[axis]):#和父节点相交，去他的右孩子找
                    _nearest_neighbor_query(node.right,depth+1)
            else:
                _nearest_neighbor_query(node.right,depth+1)#大的往右边
                if distance(node.location, query_point) < self.nearestDistance:
                    self.nearestPoint = node.location
                    self.nearestDistance = distance(node.location, query_point)
                if self.nearestDistance>abs(query_point[axis]-node.location[axis]):#和父节点相交，去他的左孩子找
                    _nearest_neighbor_query(node.left,depth+1)
        _nearest_neighbor_query(self._root)





'''
https://blog.csdn.net/qq_62940099/article/details/125671900?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125671900-blog-120048749.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125671900-blog-120048749.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=1
'''
def get_left_position(p: List[Point],axis:int=0,target:int=0)->int:
    left = 0
    right = len(p) - 1;
    while(left<right):
        mid=left+((right-left)>>1)
        if(target>p[mid][axis]):left=mid+1
        else:right=mid
    return left


def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test(n:int=1000):
    points = [Point(x, y) for x in range(n) for y in range(n)]

    lower = Point(100, 100)
    upper = Point(104, 104)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = time.perf_counter_ns()
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = time.perf_counter_ns()
    naive_time=end-start
    print(f'Naive method: {end - start}ns')

    kd = KDTree()
    start = int(round(time.time() * 1000))
    kd.insert(points)
    end = int(round(time.time() * 1000))
    print(f'build tree time: {end - start}ms')
    # k-d tree
    start = time.perf_counter_ns()
    result2 = kd.range(rectangle)
    end = time.perf_counter_ns()
    print(f'K-D tree: {(end - start)}ns')
    kd_time=end-start
    result1=sorted(result1)
    result2=sorted(result2)
    assert sorted(result1) == sorted(result2)
    return naive_time,kd_time


if __name__ == '__main__':
    #range_test()
    naive_time_set=[]
    kd_time_set=[]

    for n in range(21):
         print('n=',1000+n*100)
         naive_time,kd_time=performance_test(1000+n*100)
         naive_time_set.append(naive_time)
         kd_time_set.append(kd_time)
    # x=1
    # points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    # kd = KDTree()
    # kd.insert(points)
    # kd.nearest_neighbor_query(Point(2.1,3.1))
    # print(kd.nearestDistance,' ',kd.nearestPoint)
