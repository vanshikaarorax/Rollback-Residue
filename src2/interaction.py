import time

class InteractionLog:
    def __init__(self):
        self.events=[]

    def record(self,action,result,reward):
        self.events.append({"time":time.time(),"action":action,"result":result,"reward":reward})

    def export(self):
        return {"events":self.events}