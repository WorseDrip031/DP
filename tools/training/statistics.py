class Statistics:
    def __init__(self):
        self.values = dict()
        
    def step(self, key, value):
        total, count = 0.0, 0.0
        if key in self.values:
            total, count = self.values[key]
        total += value
        count += 1.0
        self.values[key] = total, count
        
    def get(self):
        result = dict()
        for k, (total, count) in self.values.items():
            result[k] = float(total / count)
        return result
    
    @staticmethod
    def merge(s1, s2):
        result = s1.get()
        result.update(s2.get())
        return result