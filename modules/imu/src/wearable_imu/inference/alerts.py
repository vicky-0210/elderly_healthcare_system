class AlertPolicy:
    """One alert per excursion, with a minimum alarm-to-alarm interval."""
    def __init__(self,threshold,cooldown=5.0,rearm=2):
        self.threshold,self.cooldown,self.rearm=threshold,cooldown,rearm
        self.armed=True
        self.below=0
        self.last=-float("inf")

    def step(self,score,t):
        if score<self.threshold:
            self.below+=1
            if self.below>=self.rearm: self.armed=True
            return False
        self.below=0
        if self.armed and t-self.last>=self.cooldown:
            self.armed=False
            self.last=t
            return True
        return False
