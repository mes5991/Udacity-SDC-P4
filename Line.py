import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # parameters of the last ten frames
        self.lastTenFrames = np.zeros((15,3))
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.frame = 0

    def appendNewFit(self, fit):
        index = self.frame % len(self.lastTenFrames)
        self.lastTenFrames[index] = fit
        self.frame += 1

    def getAverageFit(self):
        if self.frame < len(self.lastTenFrames) - 1:
            res = 0
            for i in range(self.frame):
                res += self.lastTenFrames[i]
            self.current_fit = res / (self.frame)
        else:
            self.current_fit = np.mean(self.lastTenFrames, axis=0)
