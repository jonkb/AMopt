""" Solution object
"""

class optsol():
    def __init__(self, x_star, f_star, g_star=None, it=-1, msg="", nfev=-1, 
            ngev=-1):
        self.x_star = x_star
        self.f_star = f_star
        self.g_star = g_star
        self.it = it
        self.msg = msg
        self.nfev = nfev
        self.ngev = ngev
    def __str__(self):
        s = "Optimization solution\n"
        s += f"\tOptimal point: x*={self.x_star}, f*={self.f_star}"
        return s
    def printall(self):
        s = "Optimization solution\n"
        s += f"\tOptimal point: x*={self.x_star}, f*={self.f_star}"
        if self.g_star is not None:
            s += f", g*={self.g_star}"
        s += f", it: {self.it}, msg: {self.msg}"
        if self.nfev > -1:
            s += f", nfev: {self.nfev}"
        if self.ngev > -1:
            s += f", ngev: {self.ngev}"
        print(s)
