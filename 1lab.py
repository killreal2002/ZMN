import sklearn.cluster as cl
import statistics as st
import random as rd
from matplotlib import pyplot as plt

num_of_elements = 50
num_of_clusters = 6

def most_common(array:list):
    temp = []
    counter = 0
    for i in range(0,len(array)):
        if(array.count(array[i])>counter):
            counter = array.count(array[i])
    for i in range(0,len(array)):
        if(array.count(array[i])==counter and array[i] not in temp):
            temp.append(array[i])
    return temp

array = []
x_array = []
y_array = []
for i in range(0,num_of_elements):
    array.append([rd.randint(0,num_of_elements),rd.randint(0,num_of_elements)])
print("array:\n",array,"\n")

for i in range(0,num_of_elements):
    x_array.append(array[i][0])
    y_array.append(array[i][1])
plt.scatter(x_array,y_array)
plt.title("Original")
plt.show()

cl_array = cl.KMeans(n_clusters=num_of_clusters,random_state=0).fit(array)
cl_labels = cl_array.labels_
cl_centers = cl_array.cluster_centers_
print("cluster_labels:\n",cl_labels,"\n")
print("cluster_centers:\n",cl_centers)

for i in range(0,num_of_clusters):
    temp = []
    temp2 = []
    temp3 = []
    for w in range(0,len(cl_labels)):
        if(cl_labels[w]==i):
            temp.append(array[w])
            temp2.append(array[w][0])
            temp3.append(array[w][1])
    print("\nCluster"+str(i)+":\n",temp)
    print("X values:",temp2)
    print("Y values:",temp3)
    print("Min X:",min(temp2))
    print("Max X:",max(temp2))
    print("Mean X:",st.mean(temp2))
    print("Median X:",st.median(temp2))
    print("Сommon X:",*most_common(temp2))
    print("Min Y:",min(temp3))
    print("Max Y:",max(temp3))
    print("Mean Y:",st.mean(temp3))
    print("Median Y:",st.median(temp3))
    print("Сommon Y:",*most_common(temp3))
    plt.scatter(temp2,temp3)
plt.title(str(num_of_clusters)+" clusters")
plt.show()

x_centers = []
y_centers = []
for i in range(0,num_of_clusters):
    x_centers.append(cl_centers[i][0])
    y_centers.append(cl_centers[i][1])
for i in range(0,num_of_clusters):
    plt.scatter(x_centers[i],y_centers[i])
plt.title(str(num_of_clusters)+" cluster centers")
plt.show()
