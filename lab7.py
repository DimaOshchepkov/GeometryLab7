from typing import List
import matplotlib.pyplot as plt
import numpy as np


class Point():
    '''
    Класс точки в 2-х мерном пространстве

    Атрибуты
    --------
        x: float
            координата по x
        y: float
            координата по y
        type: str
            тип точки
        number: int
            номер точки
    '''

    def __init__(self, x: float, y: float):
        '''
        Устанавливает все необходимые атрибуты для объекта Point

        Входные данные:
            Координаты точки x и y
        '''
        self.x: float = x
        self.y: float = y
        self.type = None
        self.number = None


    def __repr__(self) -> str:
        '''
        Возвращает строкое представление точки
        '''
        return f"({self.x}; {self.y};{self.type})"


    def __eq__(self, other) -> bool:
        '''
        Сравнение на равенство
        '''
        return self.x == other.x and self.y == other.y


    def copy(self):
        '''
        Возвращает копию экземпляра
        '''
        copy_point = Point(self.x, self.y)
        copy_point.type = self.type
        copy_point.number = self.number
        return copy_point



class Vector():
    '''
    Класс вектора в 2-х мерном пространстве

    Атрибуты
    --------
        x: float
            координата по x
        y: float
            координата по y
        begin: Point
            точка начало вектора
        end: Point
            точка конца вектора
        type: str
            тип точки
    '''

    def __init__(self, point1: Point, point2: Point):
        '''
        Устанавливает все необходимые атрибуты для объекта Vector
        '''
        self.begin: Point = point1
        self.end: Point = point2
        self.x: int = point2.x - point1.x
        self.y: int = point2.y - point1.y


    def __repr__(self) -> str:
        '''
        Вовзращает строкое представление вектора
        '''
        return f"({self.begin} -> {self.end})"
    

    def __eq__(self, other) -> bool:
        '''
        Сравнение на равенство отрезков
        '''
        reverse = Vector(other.end, other.begin)
        return (self.x == other.x and self.y == other.y and self.begin == other.begin and self.end == other.end) or \
               (self.x == reverse.x and self.y == reverse.y and self.begin == reverse.begin and self.end == reverse.end)
    

    def multiplication(self, other) -> int:
        '''
        Возвращает знак векторного произведения

        Входные параметры:
            other (Vector): второй сомножитель в векторном произведение

        Возвращаемое значение:
            Возвращается 1, если знак векторного произвдения положительный
            Возвращается -1, если знак векторного произвдения отрицательный
            Возвращается 0, если векторы параллельны
        '''
        scalar: int = self.x * other.y - self.y * other.x
        if scalar > 0:
           return 1
        if scalar < 0:
           return -1
        return 0



class Polygon():
    '''
    Класс многоугольника в 2-х мерном пространстве
    '''

    def __init__(self, list_points: List[Point]):
        self.points: List[Point] = []
        # для каждой вершины узнаем её тип, и нумеруем 
        for i in range(len(list_points)):
            # если вершина первая, то нужно знать левого соседа
            if i==0:
                list_points[i].type: str = get_type_point(list_points[-1], list_points[i], list_points[i+1])
            
            # если вершина последняя, то нужно знать соседа справа
            elif i == len(list_points)-1:
                list_points[i].type: str = get_type_point(list_points[i-1], list_points[i], list_points[0])
            
            # стандартная ситуация, вершина не первая и не последняя
            else:
                list_points[i].type: str = get_type_point(list_points[i-1], list_points[i], list_points[i+1])
            list_points[i].number = i
            self.points.append(list_points[i].copy())


    def __repr__(self) -> str:
        '''
        Вовзращает строкое представление многоугольника
        '''
        return f"{self.points}"
    

    # список из сторон многоугольника
    @property
    def sides(self) -> List[Vector]:
        '''
        Возвращает список из сторон
        '''
        list_sides: List[Vector] = []
        for i in range(len(self.points)-1):
            side = Vector(self.points[i], self.points[i+1])
            list_sides.append(side)
        side = Vector(self.points[-1], self.points[0])
        list_sides.append(side)
        return list_sides
    

    # список отсортированных точек
    @property
    def sorted_points(self) -> List[Point]:
       '''
       Возвращает отсортированный список точек по координате y
       '''
       return sorted(self.points, key=lambda point: (point.y, point.x), reverse=True) 
    
    
    # факт наличия точек merge или split
    @property
    def is_exists_merge_or_split(self) -> bool:
       '''
       Возвращает True, если точки есть
       '''
       for point in self.points:
           if point.type == "split" or point.type == "merge":
               return True
       return False
   
   

# чтение файла
def read_file(namefile: str) -> List[List[Point]]:
    '''
    Возвращает список из списков с точками

    Входные параметры:
        namefile (str): название файла

    Возвращаемое значение:
        lists_points List[List[Point]
            Список из списков
            Каждый вложенный список - точки одного многоугольника(один тестовый пример)
    '''
    with open(namefile, 'r', encoding='utf-8') as file:
        lists_points: List[List[Point]] = []
        list_str: List[str] = file.readline().strip().split()
        while list_str:
            list_points: List[Point] = [Point(float(list_str[i*2]), float(list_str[i*2+1])) for i in range(int(len(list_str)/2))]
            lists_points.append(list_points)
            list_str: List[str] = file.readline().strip().split()
    return lists_points



# отрисовка многоугольников
def rendering_polygons(list_monotonous_polygons: List[Polygon]) -> None:
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-4, 15)
    plt.ylim(-4, 15)
    plt.grid()
    
    # для каждого многоугольника
    for polygon in list_monotonous_polygons:
        X: List[float] = [] 
        Y: List[float] = []
        for point in polygon.points:
            X.append(point.x)
            Y.append(point.y)
        X.append(polygon.points[0].x)
        Y.append(polygon.points[0].y)
        plt.plot(X, Y, color='black')
    plt.show()
    


# отрисовка триангуляций
def rendering_ribs(list_triangulations: List[Vector]) -> None:
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-4, 15)
    plt.ylim(-4, 15)
    plt.grid()
    # отрисовка всех отрезков
    for vec in list_triangulations:
        plt.plot([vec.begin.x, vec.end.x], [vec.begin.y, vec.end.y], color='black')
    plt.show()



# обход точек
def is_clockwise_walk(A: Point, B: Point, C: Point) -> bool:
    '''
    Определение обхода A→B→C:
      Обход по часовой стрелке или против часовой стрелки.

    Возврат:
      True, если поворот по часовой стрелке
    '''
    if ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y) < 0):
      return True
    return False



# получение типа точки
def get_type_point(A: Point, B: Point, C: Point) -> str:
    # если B выше всех
    if (B.y > A.y or (B.y==A.y and B.x < A.x)) and \
       (B.y > C.y or (B.y==C.y and B.x < C.x)):
       # угол острый = start
       if is_clockwise_walk(A, B, C):
           return "start"
       # угол тупой = split
       else:
           return "split"
    # если B ниже всех
    elif (B.y < A.y or (B.y==A.y and B.x > A.x)) and \
         (B.y < C.y or (B.y==C.y and B.x > C.x)):
         # угол острый = end 
         if is_clockwise_walk(A, B, C):
            return "end"
         # угол острый = merge  
         else:
             return "merge"
    # иначе = regular
    else:
        return "regular"



# лежит ли точка B на AC
def is_point_middle(A: Point, B: Point, C: Point) -> bool:
    '''
    Определяет, находится ли точка B между A и С

    Входные данные:
       3 точки, представленные классом Point

    Выходные данные:
        True, если B лежит на отрезке AC

    Примечание:
       Исходные точки должны быть на одной прямой
    '''
    # если B - это или A, или С, то пересечения нет, в вершинах не считается
    if (B==C) or (B==A) or (A==C):
        return False

    AB = Vector(A, B)
    AC = Vector(A, C)

    #если произведение соответствующих коорд больше 0, то векторы сонаправлены
    if AB.x * AC.x >= 0 and AB.y * AC.y >=0:
        #смотрим длинны сонаправленных векторов, где начало векторов - одна и таже точка
        #если надо чтобы векторы отличались по длине
        #при >= есть шанс, что векторы сопадут и две точки окажутся с одинаковыми координатами
        if (abs(AB.x) > abs(AC.x) and abs(AB.y) >= abs(AC.y)) or \
            (abs(AB.x) >= abs(AC.x) and abs(AB.y) > abs(AC.y)):
                return False #С - посередине
        else:
            return True #B - посередине

    #иначе вектора разнонаправленные, и выходят из одной точки (A - посередине)
    else:
        return False
    


# факт пересечения или непересечения отрезков
def is_intersection_of_segments(segment1: Vector, segment2: Vector) -> bool:
    '''
    Определяет - имеют ли отрезки общие точки

    Входные данные:
        Два отрезка, представлены классом Vector, хранящим начало и конец отрезка

    Выходные данные:
        Возвращает True, если отрезки пересекаются НЕ в вершинах
    '''
    p1: Point = segment1.begin
    p2: Point = segment1.end

    p3: Point = segment2.begin
    p4: Point = segment2.end

    v1: int = Vector(p3, p4).multiplication(Vector(p3, p1))
    v2: int = Vector(p3, p4).multiplication(Vector(p3, p2))
    v3: int = Vector(p1, p2).multiplication(Vector(p1, p3))
    v4: int = Vector(p1, p2).multiplication(Vector(p1, p4))

    # Отрезки пересекаются
    if v1*v2<0 and v3*v4<0:
        return True

    # Отрезки не пересекаются
    if v1*v2>0 or v3*v4>0:
        return False

    # Точка р3 лежит на отрезке р1р2
    if v1*v2<=0 and v3==0 and v4!=0:
        return False

    # Точка р4 лежит на отрезке р1р2
    if v1*v2<=0 and v4==0 and v3!=0:
        return False

    # Точка р1 лежит на отрезке р3р4
    if v3*v4<=0 and v1==0 and v2!=0:
        return False

    # Точка р2 лежит на отрезке р3р4
    if v3*v4<=0 and v2==0 and v1!=0:
        return False

    # Отрезки р1р2 и р3р4 лежат на одной прямой
    if v1==v2==v3==v4==0:
        if is_point_middle(p1, p3, p2) or \
           is_point_middle(p1, p4, p2) or \
           is_point_middle(p3, p1, p4) or \
           is_point_middle(p4, p2, p4):
                return True
    return False



# проверка пересечения ребра со сторонами многоугольника
def intersection(ribs: Vector, list_vectors: List[Vector]) -> bool:
    '''
    Выходные данные:
        Возвращает True, если потенциальное ребро триангуляции имеет пересечение со стороной многоугольника
    '''
    # смотрим пересечение каждой строны с потенциальным ребром
    for vector in list_vectors:
        if vector == ribs:
            return True
        elif is_intersection_of_segments(ribs, vector):
            return True
    return False



def is_intersection_ray_and_segment(A: Point, k1: float, b1: float, list_points: List[Point]) -> int:
    '''
    Определяет как пересекается луч и сторона многоугольника(отрезок)

    Входные данные:
        A: Point
            точка, из котороый выходит луч
        k1: float
            коэф k луча(прямой)
        b1: float
            коэф b луча(прямой)
        list_points: List[Point]
            Список из двух точек, определяющих сторону многоугольника(отрезок)

    Выходные данные:
        Возвращает 1, если есть "ХОРОШЕЕ" пересечние луча с отрезком
        Возвращает 0, если нет пересечений
        Возвращает -1, если пересечение прроисходит на границах отрезка или отрезок лежит на луче

    Примечания:
        "хорошее пересечение" - пересечение происходит НЕ в концах отрезка И
                   луч и отрезок лежат НЕ на одной прямой
    '''
    #вершины стороны многоульника
    X1: Point = list_points[0]
    X2: Point = list_points[1]

    #если обе точки стороны многоугольника по одну сторону от луча
    if (X1.y > k1*X1.x+b1 and X2.y > k1*X2.x+b1) or \
        (X1.y < k1*X1.x+b1 and X2.y < k1*X2.x+b1):
        return 0

    #если точки отрезка по разные стороны от луча
    elif (X1.y > k1*X1.x+b1 and X2.y < k1*X2.x+b1) or \
        (X1.y < k1*X1.x+b1 and X2.y > k1*X2.x+b1):
        # если отрезок (сторона многоугольника) вертикальный
        if X1.x == X2.x:
            # если сторона многоугольника справа, то точка внутри многоугольника(луч от A идёт изначально вправо)
            if X1.x > A.x:
                return 1
            else:
                return 0
        # если отрезок не вертикальный
        else:
            # угловой коэф. прямой отрезка(сторона многоугольника)
            k2: float = (X2.y-X1.y) / (X2.x-X1.x)
            # свободный член прямой отрезка
            b2: float = X2.y - X2.x*k2
            # точка пересечения луча и отрезка k1*x+b1=k2*x+b2
            k = k1 - k2
            b = b2 - b1
            #координаты х и y точки пересечения
            x: float = b / k
            y: float = k1 * x + b1
            #если координата точки пересечения справа от точки А, то точка А внутри
            if x > A.x:
                return 1
            else:
                return 0

    #если прямая параллельна отрезку или попала в край отрезка то возвращаем -1
    elif (X1.y == k1*X1.x+b1 or X2.y == k1*X2.x+b1):
        return -1
    

# проверка - внутри или снаружи ребро
def is_inner_ribs(ribs: Vector, polygon_points: List[Point]) -> bool:
    '''
    Определяет находится точка внутри многоугольника

    Входные данные:
        Список из списков точек:
            1 подсписок содержит одну точку проверяемой точки
            2 подcписок содержит точки многоугольника в отсортированном порядке

    Выходные данные:
        Возвращает True, если точка входит в многоугольник
    '''
    # точка A, проверяемая на вхождение в многоугольник
    A: Point = Point((ribs.begin.x + ribs.end.x)/2, (ribs.begin.y + ribs.end.y)/2)

    # общее количество точек многоугольника
    count_point = len(polygon_points)

    # начальные координаты луча(горизонатальный), который будет проводится от точки A под углом 0 градусов
    # луч будет поворачиваться под определённым углом (30)
    x_left: float = -10.0
    x_right: float = 10.0
    y_left: float = 5
    y_right: float = 5

    # как только попадётся хороший луч, т.е.
    # не лежит на стороне многоугольника и не пересекается в вершине многоугольника
    for degree in range (0, 360, 30):
        # перевод градусов в радианы
        radian: float = np.pi/180*degree

        # поворот луча, находим новые координаты луча
        # x' = x*cos - y*sin      y' = x*sin + y*cos

        # проверка при преобразовании новых координат в старой системе координат
        if ((x_right*np.sin(radian)-y_right*np.sin(radian))-(x_left*np.sin(radian)-y_left*np.sin(radian))) != 0:

            # преобразование прямой поворотом, находим линейный коэф прямой
            k: float = ((x_right*np.cos(radian)+y_right*np.cos(radian))-(x_left*np.cos(radian)+y_left*np.cos(radian))) \
                        /((x_right*np.sin(radian)-y_right*np.sin(radian))-(x_left*np.sin(radian)-y_left*np.sin(radian)))

            # находим b, такое чтобы точка A лежала на луче
            b: float = A.y - A.x * k

            #кол-во пересечений луча со сторонами многоугольника
            count: int = 0

            # флаг, подходит ли луч для решения задачи
            is_true: bool = True

            #перебор всех сторон на пересечение с лучом
            for j in range(count_point):

                # список их двух точек, представляющих сторону
                temp_list_points: List[Point] = []

                # если сторона последняя, то берём 1 вершину и последнюю
                if j == count_point - 1:
                    temp_list_points.append(polygon_points[-1])
                    temp_list_points.append(polygon_points[0])

                # иначе берём сторону из последовательных отсортированных точек
                else:
                    temp_list_points.append(polygon_points[j])
                    temp_list_points.append(polygon_points[j+1])

                # проверяем пересечения луча и стороны многоугольника
                is_intersection: int = is_intersection_ray_and_segment(A, k, b, temp_list_points)

                # если пересечение есть и оно хорошее (не в вершине и не параллельность луча со стороной)
                if is_intersection == 1:
                    count += 1
                # пересечение плохое, меняем угол луча
                elif is_intersection == -1:
                    is_true = False
                    break

            # если луч хороший
            if (is_true):
                #если нечетное кол-во пересечений, то точка внутренняя
                if count % 2 == 1:
                    return True
                #если четное кол-во пересечений, то точка внешняя
                else:
                    return False



# получение двух монотонных многоугольников из одного большого
def get_monotonous_polygons(polygon: Polygon) -> List[Polygon]:
    
    # отсортированные точки по 'y' сверху вниз
    sorted_points: List[Point] = polygon.sorted_points
    
    # индексы вершин, которые будут соединятся, для разбиения многоугольника на монотонные
    index_merge_or_split_point: int = -1
    index_connected_point: int = -1
    
    # перебор отсортированных точек
    for i in range(len(sorted_points)):
        
        # если merge вершина
        if sorted_points[i].type == "merge":
            
            # для каждой точки, которые ниже данной merge вершины
            for j in range(i+1, len(sorted_points)):
                
                # находим ребро от merge
                ribs = Vector(sorted_points[i], sorted_points[j])
                
                # проверяем ребро на пересечение со сторонами многоугольника
                # ребро внутри
                if not intersection(ribs, polygon.sides) and is_inner_ribs(ribs, polygon.points):
                    
                    # индексы вершин, которые будут соединятся, для разбиения многоугольника на монотонные
                    index_merge_or_split_point: int = sorted_points[i].number
                    index_connected_point: int = sorted_points[j].number
                    print(sorted_points[i])
                    break
                
        # если split вершина
        elif sorted_points[i].type == "split":
            
            # для каждой точки, которые выше данной merge вершины
            for j in range(i-1, -1, -1):

                # находим ребро от split
                ribs = Vector(sorted_points[i], sorted_points[j])

                # проверяем ребро на пересечение со сторонами многоугольника
                # ребро внутри
                if not intersection(ribs, polygon.sides) and is_inner_ribs(ribs, polygon.points):
                    
                   # индексы вершин, которые будут соединятся, для разбиения многоугольника на монотонные
                    index_merge_or_split_point: int = sorted_points[i].number
                    index_connected_point: int = sorted_points[j].number
                    print(sorted_points[i])
                    break
    
    # делим многоугольник
                
    # индексы двух точек, по которым делится многоугольник
    min_index_point: int = min(index_merge_or_split_point, index_connected_point)
    max_index_point: int = max(index_merge_or_split_point, index_connected_point)

    # 1 многоугольник
    list_point_polygon1: List[Point] = []
    # копируем точки
    for i in range(min_index_point, max_index_point+1):
        list_point_polygon1.append(polygon.points[i].copy())
    print(list_point_polygon1)
    polygon1 = Polygon(list_point_polygon1)

    # 2 многоугольника
    list_point_polygon2: List[Point] = []
    # копируем точки
    for i in range(max_index_point, len(polygon.points)):
        list_point_polygon2.append(polygon.points[i].copy())
    for i in range(min_index_point+1):
        list_point_polygon2.append(polygon.points[i].copy())
    print(list_point_polygon2)
    polygon2 = Polygon(list_point_polygon2)
    
    return [polygon1, polygon2]



# получение списка монотонных многоугольников
def get_list_monotonous_polygon(original_polygon: Polygon) -> List[Polygon]:
    
    # флаг, что есть ещё точки split и merge
    is_exists_point_merge_and_split: bool = True
    
    # лист из монотонных многоугольников
    list_monotonous_polygons: List[Polygon] = [original_polygon]
    
    #пока есть точки, значит делим многоугольник на несколько многоугольников
    while is_exists_point_merge_and_split:
        
        # для каждого многоугольника смотрим наличие точек split и merge
        for i in range(len(list_monotonous_polygons)):
            
            # если точки есть, то делим многоугольник
            if list_monotonous_polygons[i].is_exists_merge_or_split:
                
                # находим два новых многоугольника
                new_list_monotonous_polygons: List[Polygon] = get_monotonous_polygons(list_monotonous_polygons[i])
                
                # удаляем стырый многоугольник
                list_monotonous_polygons.pop(i)
                
                # добавялем два новых
                list_monotonous_polygons.extend(new_list_monotonous_polygons)
                break
            
        # если из цикла не вышли через break, значит нет вершин merge и split, разбили всё на монотонные
        else:
            is_exists_point_merge_and_split = False
          
    return list_monotonous_polygons



# получение триангуляций для монотонных многоугольников
def get_list_triangulations(list_monotonous_polygons: List[Polygon]) -> List[Vector]:
    # список со всеми триангуляциями
    list_all_triangulations: List[Vector] = []
    
    # для каждого монотонного многоугольника
    for polygon in list_monotonous_polygons:
        # список из проведённых ребёр одного монотонного многоугольника
        list_triangulations: List[Vector] = []
        
        # необходимое количество рёбер для многоугольника
        
        need_count_ribs: int = len(polygon.points) - 3
        # пока провели не все рёбра, то проводим
        
        while (need_count_ribs > 0):
            
            # для двух разных точек
            for i in range(len(polygon.points)):
                for j in range(len(polygon.points)):
                    if i!=j:
                        # формируем потенциальное ребро
                        ribs = Vector(polygon.points[i], polygon.points[j])
                        
                        # проверяем, чтобы оно было внутри и не пересекалось с другими рёбрами
                        if not intersection(ribs, polygon.sides + list_triangulations) and is_inner_ribs(ribs, polygon.points):
                           
                            # индексы вершин, которые будут соединятся, для разбиения многоугольника на монотонные
                            list_triangulations.append(ribs)
                            need_count_ribs -= 1
                            if need_count_ribs==0:
                                break
                if need_count_ribs==0:
                    break
                
        # добавляем рёбра для отрисовки
        list_all_triangulations.extend(list_triangulations)
        
        # добавляем стороны многоугольника для отрисовки
        list_all_triangulations.extend(polygon.sides)
    return list_all_triangulations


# поиск триангуляций
def zd1(list_points: List[Point]) -> None:
    # для списка точек формируем экземпляр класса Polygon
    original_polygon = Polygon(list_points)
   
    # отрисовка изначального многоульника
    rendering_polygons([original_polygon])

    # поиск монотонных многоугольников
    list_monotonous_polygons: List[Polygon] = get_list_monotonous_polygon(original_polygon)
    
    # отрисовка монотонных многоугольников
    rendering_polygons(list_monotonous_polygons)
   
    # разбиение на триангуляции всех монотонных многоугольников
    list_triangulations: List[Vector] = get_list_triangulations(list_monotonous_polygons)
    
    # отрисовка ответа
    rendering_ribs(list_triangulations)



def main() -> None:
    # список из списков с точками
    lists_points: List[List[Point]] = read_file("input1.txt")
    
    # для каждого списка с точками находим триангуляцию 
    for list_points in lists_points:
        zd1(list_points)



if __name__=="__main__":
    main()