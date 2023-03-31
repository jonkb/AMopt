import numpy as np
from obj2_func import obj_func

#search all files in directory to find the best solution
def search():
    i = 5000
    best_f = 1e10
    best_x = np.zeros(125)
    best_i = 0
    bests = 0
    while i <= 5445:
        #read the file
        x = np.loadtxt(f'x00{i}.txt')
        g = np.loadtxt(f'x00{i}_max_stress.txt', dtype=float)
        #calculate the objective function
        f = obj_func(x)
        #check if the solution is feasible
        if g <= 50:
            #check if the solution is better than the previous best solution
            if f < best_f:
                #update the best solution
                best_f = f
                best_x = x
                best_i = i
                bests += 1
                
        i += 1

    # print the best solution
    print(best_f)
    print(best_x)
    print(best_i)
    print(bests)

def compile_warm_start():
    #write the last 1250 text files to a 1250x125 matrix
    x = np.zeros((1250,125))
    for i in range(1250):
        x[i] = np.loadtxt(f'x00{i+4195}.txt')
    #write the matrix to a text file
    np.savetxt('warm_start5x5x5.txt', x)
    print('done')

if __name__ == '__main__':
    # search()
    compile_warm_start()
