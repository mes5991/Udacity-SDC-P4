import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # parameters of the last ten frames
        self.lastTenFrames = np.zeros((15,3))
        #average x values of the fitted line over the last n iterations
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.frame = 0

    def appendNewFit(self, fit):
        #Get proper index in self.lastTenFrames to replace with new polynomial
        index = self.frame % len(self.lastTenFrames)
        #Replace old polynomial with new polynomial
        self.lastTenFrames[index] = fit
        #Update frame count
        self.frame += 1

    def getAverageFit(self):
        #If the system hasnt filled self.lastTenFrames yet, average the polynomial manually
        if self.frame < len(self.lastTenFrames) - 1:
            res = 0
            for i in range(self.frame):
                res += self.lastTenFrames[i]
            self.current_fit = res / (self.frame)
        else:
            #If the system has already filled self.lastTenFrames, average the polynomials using np.mean
            self.current_fit = np.mean(self.lastTenFrames, axis=0)
