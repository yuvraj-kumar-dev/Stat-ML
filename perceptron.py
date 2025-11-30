import numpy as np

class perceptron:

    """
        features : n*m array with n rows and m columns
        target : 1d array
        threshold : int (for activation function)

        ex: for threshold = 0 {if value >= 0 : return 1 else : 0}

    """

    def __init__(self, features, target, threshold):
        self.features = np.array(features)
        self.target = np.array(target)
        self.threshold = threshold

    def fit(self, lr, epochs):
        w = np.zeros(len(self.features[0]))
        b = 0

        for j in range(epochs):

            for i in range(len(self.target)):
                net = np.dot(w,self.features[i])+b
                if net >= self.threshold:
                    pred = 1
                else:
                    pred = 0
                error = self.target[i] - pred
                w = w + lr*(error)*self.features[i]
                b = b + lr*error
        self.w = w
        self.b = b
        return self.w, self.b
    
    def predict(self,feat):
        pred = np.dot(self.w, feat)+self.b
        if pred >= 0: 
            return 1
        else:
            return 0


# Testing the perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

feat = np.array([1,0])

model = perceptron(X,y,threshold=0)
print(model.fit(lr=0.1,epochs=5)) # (array([0.2, 0.1]), np.float64(-0.20000000000000004))
print(model.predict(feat)) # 0


    



        