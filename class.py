"""
Algoritmo Constructivo Problema Vehicle
Routing Problem with Time Windows – VRPTW
Clase
Juan Fernando Riascos Goyes
"""
class Nodo:
    def __init__(self,index,x_cord,y_cord,demand,inflim,suplim,serv):
       self.index = index 
       self.x_cord = x_cord
       self.y_cord = y_cord
       self.demand = demand
       self.inflim = inflim
       self.suplim = suplim
       self.serv = serv
    
    
    def __index__(self):
        return self.index
# Output: n  
    def __hash__(self):
        return self.index
# Output: Customer <n>
    def __repr__(self):
        return f"Customer <{self.index}>"
