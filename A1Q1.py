import numpy as np
import matplotlib.pyplot as plt

inti_Power = 1 #initial transmit powers
noise = 0.1 #noise
iteration_num = 10 #iterations

iteration_record_Pwr = []
iteration_record_SIR = []
matrix_G = np.array([[1,0.1,0.3],[0.2,1,0.3],[0.2,0.2,1]], dtype = np.float_)
Gamma = np.array([[1],[1.5],[1]], dtype = np.float_)


num_Reciever = matrix_G.shape[1]
matrix_SIR = np.zeros((num_Reciever, 1), dtype = float)
matrix_Pwr = np.full((num_Reciever, 1), inti_Power, dtype = float)


def calculate_SIR(mat_G, mat_SIR, mat_P):
    for i in range(num_Reciever):
        sum_interference = 0
        for j in range(num_Reciever):
            if i != j:
                sum_interference += mat_G[i][j]*mat_P[i][0]
        mat_SIR[i][0] = mat_P[i][0]*mat_G[i][i]/(sum_interference + noise)
    return 0

def DPC_iter(mat_SIR, mat_P, mat_Gm):
    for i in range(num_Reciever):
        mat_P[i][0] = (mat_Gm[i][0]/mat_SIR[i][0])*mat_P[i][0]
    return mat_P

def data_reshape(list_matrix):
    reshaped_list = []
    for i in range(num_Reciever):
        reshaped_list.append([])
        for items in list_matrix:
            reshaped_list[i].append(items[i][0])
    
    return reshaped_list


iteration_record_SIR.append(matrix_SIR)
iteration_record_Pwr.append(matrix_Pwr)
for i in range(iteration_num):
    calculate_SIR(matrix_G, matrix_SIR, matrix_Pwr)
    DPC_iter(matrix_SIR, matrix_Pwr, Gamma)
    iteration_record_SIR.append(matrix_SIR)
    iteration_record_Pwr.append(matrix_Pwr)
    #print(matrix_SIR)

plot_list_SIR = data_reshape(iteration_record_SIR)
plot_list_Pwr = data_reshape(iteration_record_Pwr)

# print(plot_list_SIR)
# print(matrix_Pwr)

x = range(iteration_num + 1)
fig_SIR = plt.figure()
plt.title("SIR")
for i in range(num_Reciever):
    plt.plot(x, plot_list_SIR[i])
fig_SIR.savefig("C:\\Users\\stanchen\\Documents\\Code\\temp.png")


