import numpy as np
import random as rd
import time

N = 264
M = 264
numafterdecimal = 2

def vector_mult(matrix_A,matrix_B,A_rows,B_rows,B_columns):
    matrix_C = []
    for i in range(0,A_rows):
        temp = []
        for j in range(0,B_columns):
            for k in range(0,B_rows):
                temp.append(round(matrix_A[i][k]*matrix_B[k][j],numafterdecimal))
        matrix_C.append(temp)
    return matrix_C

def matrix_mult_vector(matrix_A,matrix_B,A_rows,B_rows,B_columns):
    matrix_C = []
    for i in range(0,A_rows):
        temp = []
        for j in range(0,B_columns):
            temp2 = 0
            for k in range(0,B_rows):
                temp2+=round(matrix_A[i][k]*matrix_B[k][j],numafterdecimal)
            temp.append(round(temp2,numafterdecimal))
        matrix_C.append(temp)
    return matrix_C

Npy_array = []
Mpy_array = []
                                                                       # [rows]x[columns]
for i in range(0,N):
    Npy_array.append([round(rd.random(),numafterdecimal)])             # initializing Npy [100][1]
print("Npy_array("+str(N)+"x1):\n",Npy_array,"\n")
temp = []
for i in range(0,M):
    temp.append(round(rd.random(),numafterdecimal))
Mpy_array.append(temp)                                                 # initializing Mpy [1][100]
print("Mpy_array(1x"+str(M)+"):\n",Mpy_array,"\n")
############################################################
Nnum_array = np.array(Npy_array)
Mnum_array = np.array(Mpy_array)
print("Nnum_array("+str(N)+"x1):\n",Nnum_array,"\n")
print("Mnum_array(1x"+str(M)+"):\n",Mnum_array,"\n")
############################################################
start = time.time()
N_mult_M = vector_mult(Npy_array,Mpy_array,N,1,M)                      # Npy*Mpy matrix
end = time.time()
time_result11 = round(end-start,numafterdecimal)
print("PY N*M:\n",N_mult_M,"\n","time:",time_result11,"\n")
start = time.time()
NmultM = np.matmul(Nnum_array,Mnum_array)
end = time.time()
time_result12 = round(end-start,numafterdecimal)
print("NumPY N*M:\n",NmultM,"\n","time:",time_result12,"\n")
############################################################
start = time.time()
N_mult_M_mult_N = matrix_mult_vector(N_mult_M,Npy_array,N,N,1)         # (Npy*Mpy)*Npy matrix
end = time.time()
time_result21 = round(end-start,numafterdecimal)
print("PY (N*M)*N:\n",N_mult_M_mult_N,"\n","time:",time_result21,"\n")
start = time.time()
NmultMmultN = np.matmul(NmultM,Nnum_array)
end = time.time()
time_result22 = round(end-start,numafterdecimal)
print("NumPY (N*M)*N:\n",NmultMmultN,"\n","time:",time_result22,"\n")
############################################################
start = time.time()
M_mult_N_mult_M = matrix_mult_vector(Mpy_array,N_mult_M,1,N,M)         # Mpy(Npy*Mpy) matrix
end = time.time()
time_result31 = round(end-start,numafterdecimal)
print("PY M*(N*M):\n",M_mult_N_mult_M,"\n","time:",time_result31,"\n")
start = time.time()
MmultNmultM = np.matmul(Mnum_array,NmultM)
end = time.time()
time_result32 = round(end-start,numafterdecimal)
print("NumPY M*(N*M):\n",MmultNmultM,"\n","time:",time_result32,"\n")
############################################################
start = time.time()
N_mult_M_mult_N_mult_M = matrix_mult_vector(N_mult_M,N_mult_M,N,N,M)   # (Npy*Mpy)*(Npy*Mpy)
end = time.time()
time_result41 = round(end-start,numafterdecimal)
print("PY (N*M)*(N*M):\n",N_mult_M_mult_N_mult_M,"\n","time:",time_result41,"\n")
start = time.time()
NmultMmultNmultM = np.matmul(NmultM,NmultM)
end = time.time()
time_result42 = round(end-start,numafterdecimal)
print("NumPY (N*M)*(N*M):\n",NmultMmultNmultM,"\n","time:",time_result42,"\n")
############################################################
unit_test_1_array1 = np.array([[1],[2],[3]])
unit_test_1_array2 = np.array([[1,2,3]])
print("Unit Test 1 The answer of NumPY:\n",np.matmul(unit_test_1_array1,unit_test_1_array2))
assert vector_mult([[1],[2],[3]],[[1,2,3]],3,1,3) == [[1,2,3],[2,4,6],[3,6,9]]
print("Unit Test 1 for vector_mult() Passed")
############################################################
unit_test_2_array1 = np.array([[1,2,3],[2,4,6],[3,6,9]])
unit_test_2_array2 = np.array([[1],[2],[3]])
print("Unit Test 2 The answer of NumPY:\n",np.matmul(unit_test_2_array1,unit_test_2_array2))
assert matrix_mult_vector([[1,2,3],[2,4,6],[3,6,9]],[[1],[2],[3]],3,3,1) == [[14],[28],[42]]
print("Unit Test 2 for matrix_mult_vector() Passed")
############################################################
print("PY N*M time: ",time_result11, " NumPy N*M time: ",time_result12)
print("PY (N*M)*N time: ",time_result21, " NumPy (N*M)*N time: ",time_result22)
print("PY M*(N*M) time: ",time_result31, " NumPy M*(N*M) time: ",time_result32)
print("PY (N*M)*(N*M) time: ",time_result41, " NumPy (N*M)*(N*M) time: ",time_result42)
