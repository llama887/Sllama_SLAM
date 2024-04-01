import deadreckoning as dr
import place_recognition as pr

class Dijstraka():
    def __init__(self):
        self.unvisited = []
        self.costs = {}
        self.parents = {}
        self.visited = []
        self.directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1, -1), (-1, 0), (-1, 1)]
        self.t_map = None
        self.goal_pos = None

    def search(self, pos_tuple, localizer_map):
        #breakpoint()
        print(f"Searching...")
        curr_pos_x = 0
        curr_pos_y = 0
        curr_pos = (curr_pos_x, curr_pos_y)
        self.goal_pos = (pos_tuple[0], pos_tuple[1])
        if self.target_found(curr_pos):
            print("Goal is start!")
            return None, None
        curr_cost = 0
        self.t_map = list(zip(localizer_map.x, localizer_map.y))
        self.unvisited.append(curr_pos)
        self.costs[curr_pos] = curr_cost
        self.parents[curr_pos] = None
        while len(self.unvisited) > 0:
            curr_pos = self.unvisited.pop()

            curr_pos_x = curr_pos[0]
            curr_pos_y = curr_pos[1]
            curr_cost = self.costs[curr_pos]

            self.visited.append(curr_pos)

            for dir in self.directions:
                next_pos_x = curr_pos_x + dir[0]
                next_pos_y = curr_pos_y + dir[1]
                next_pos = (next_pos_x, next_pos_y)
                if self.valid_node(next_pos):
                    if self.target_found(next_pos):
                        self.parents[next_pos] = curr_pos
                        print("Goal is found!")
                        mapped_x, mapped_y = self.trace_path()
                        return mapped_x, mapped_y
                    else:
                        next_cost = curr_cost + 1
                        if self.new_unvisited(next_pos, next_cost):
                            self.unvisited.append(next_pos)
                            self.costs[next_pos] = next_cost
                            self.parents[next_pos] = curr_pos
        return None, None
                            
    def in_map(self, pos):
        if pos in self.t_map:
            return True
        return False
    def is_visited(self, pos):
        if pos in self.visited:
            return True
        return False 

    def is_unvisited(self, pos):
        if pos in self.unvisited:
            return True
        return False

    def lesser_cost(self, pos, cost):
        if self.costs[pos] > cost:
            return True
        return False

    def new_unvisited(self, pos, cost):
        if not self.is_unvisited(pos) or self.lesser_cost(pos, cost):
            return True
        return False

    def valid_node(self, pos):
        if self.in_map(pos) and not self.is_visited(pos):
            return True
        return False   

    def target_found(self,curr_pos):
        if curr_pos == self.goal_pos:
            return True
        return False
    
    def trace_path(self):
        curr_pos = self.goal_pos
        mapped_x = []
        mapped_y = []
        while(True):
            curr_pos = self.parents[curr_pos]
            if curr_pos != None:
                curr_pos_x = curr_pos[0]
                curr_pos_y = curr_pos[1]
                mapped_x.append(curr_pos_x)
                mapped_y.append(curr_pos_y)
            else:
                break
        return mapped_x, mapped_y
    