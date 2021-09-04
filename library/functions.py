from os import umask
import numpy as np
import pandas as pd

#### pads the given array in matrix form and create a pad to simulate Neumann BC ####
def pad(args):
    return np.pad(args,((1,1), (1,1)), mode = 'edge')


#### Reshapes the given matrix to a column of samples to be inputted to the model or tested####
#   It goes thru the matrix like:  (vertical-wise)

def reshape_U(arr,n=100):
    #print("before padding shape", arr.shape)
    
    padded_arr = np.pad(arr,((1,1), (1,1)), mode = 'edge')
    #print("after padding shape", padded_arr.shape)
    U = np.zeros(((n+1)*(n+1),5))
    i = 0
    for y_point in range(1,n+2):
        for x_point in range(1,n+2):
                U[i,0] = padded_arr[x_point,y_point] #i,j   point
                U[i,1] = padded_arr[x_point-1,y_point] #i,j+1 point
                U[i,2] = padded_arr[x_point,y_point+1] #i+1,j point
                U[i,3] = padded_arr[x_point+1,y_point] #i,j-1 point
                U[i,4] = padded_arr[x_point,y_point-1] #i-1,j point 
                i += 1
    # print ("U shape = ", U.shape)
    return U

### Computes Ut from U(t) & U(t-1) to compute the loss. Horizontal-wise ###

def get_Ut(arr1,arr2, n=100):

    Ut = np.zeros((n+1)*(n+1))
    i = 0
    for y_point in range(0,n+1):
            for x_point in range(0,n+1):
                    Ut[i] = arr2[x_point, y_point]     -   arr1[x_point, y_point ] #i,j   point
                    i += 1

    #print("Ut shape = ", Ut.shape)
    return Ut


### Import data from Fenics

def get_FEniCSdata(f_name="f60604", n=250):
    
    titles = ["Points:0","Points:1","f_868"]

    fenics_t1 = (pd.read_csv("fenics_files/heat_eqn_0.csv" , delimiter = ",").reindex(columns = titles)).to_numpy()

    fenics_t2 = (pd.read_csv("fenics_files/heat_eqn_1.csv", delimiter = ",").reindex(columns = titles)).to_numpy()

    #fenics_tst = (pd.read_csv("fenics_files/tstindex.csv", delimiter = ",").reindex(columns = titles)).to_numpy()


    transformation = np.reshape(fenics_t1[:,2],(n+1,n+1))
    transformation = np.transpose(transformation)
    #print(transformation[0:15,0:5])
    U_fenics = reshape_U(transformation,n)

    Ut_fenics = fenics_t2[:,2]-fenics_t1[:,2]
    #Ut_fenics = fenics_tst[:,2]



    print("Ut shape = ", Ut_fenics.shape)
    print("U shape = ", U_fenics.shape)

       
    return [U_fenics, Ut_fenics]


### Randomize unision

def randomize_unision(arr1, arr2):

    shuffler = np.random.permutation(len(arr1))
    U_test= arr1[shuffler]
    Ut_test = arr2[shuffler]
    return U_test, Ut_test






def LoadFEniCS_andAugmentData():

    titles = ["Points:0","f_46576"]

    fenics_fp0= (pd.read_csv("fenics_files/fokkerplank_0.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp1= (pd.read_csv("fenics_files/fokkerplank_1.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp2= (pd.read_csv("fenics_files/fokkerplank_2.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp3= (pd.read_csv("fenics_files/fokkerplank_3.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp4= (pd.read_csv("fenics_files/fokkerplank_4.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp5= (pd.read_csv("fenics_files/fokkerplank_5.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp6= (pd.read_csv("fenics_files/fokkerplank_6.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp7= (pd.read_csv("fenics_files/fokkerplank_7.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp8= (pd.read_csv("fenics_files/fokkerplank_8.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp9= (pd.read_csv("fenics_files/fokkerplank_9.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp10= (pd.read_csv("fenics_files/fokkerplank_10.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp11= (pd.read_csv("fenics_files/fokkerplank_11.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp12= (pd.read_csv("fenics_files/fokkerplank_12.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp13= (pd.read_csv("fenics_files/fokkerplank_13.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp14= (pd.read_csv("fenics_files/fokkerplank_14.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp15= (pd.read_csv("fenics_files/fokkerplank_15.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp16= (pd.read_csv("fenics_files/fokkerplank_16.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp17= (pd.read_csv("fenics_files/fokkerplank_17.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp18= (pd.read_csv("fenics_files/fokkerplank_18.csv" , delimiter = ",").reindex(columns = titles).to_numpy())
    fenics_fp19= (pd.read_csv("fenics_files/fokkerplank_19.csv" , delimiter = ",").reindex(columns = titles).to_numpy())

    U_fp1 = shape_fp(fenics_fp1)
    U_fp2 = shape_fp(fenics_fp2)
    U_fp3 = shape_fp(fenics_fp3)
    U_fp4 = shape_fp(fenics_fp4)
    U_fp5 = shape_fp(fenics_fp5)
    U_fp6 = shape_fp(fenics_fp6)
    U_fp7 = shape_fp(fenics_fp7)
    U_fp8 = shape_fp(fenics_fp8)
    U_fp9 = shape_fp(fenics_fp9)
    U_fp10 = shape_fp(fenics_fp10)
    U_fp11 = shape_fp(fenics_fp11)
    U_fp12 = shape_fp(fenics_fp12)
    U_fp13 = shape_fp(fenics_fp13)
    U_fp14 = shape_fp(fenics_fp14)
    U_fp15 = shape_fp(fenics_fp15)
    U_fp16 = shape_fp(fenics_fp16)








    U = np.append(U_fp_append1,U_fp_append2,axis=0)
    Ut = np.append(Ut_fp_append1 ,Ut_fp_append2,axis=0)




    # Shape to input NN
def shape_fp(arrr):
    padded_arr = np.pad(arrr[:,1],(2,2),  'edge')
    n = arrr.shape[0]
    U = np.zeros((n,3))
    i = 0
    for x_point in range(2,n+2):
            U[i,0] = padded_arr[x_point] #i,j   point
            U[i,1] = padded_arr[x_point-1] #i,j+1 point
            U[i,2] = padded_arr[x_point-2] #i+1,j point
            U[i,3] = padded_arr[x_point+1] #i,j+1 point
            U[i,2] = padded_arr[x_point+1] #i+1,j point
            i+= 1
    return U



def linearcombi(U_fp1, U_fp2,U_fp3,U_fp4,U_fp5,U_fp6,U_fp7,U_fp8,U_fp9):
    
    
    Ut_fp1 = U_fp1 - U_fp2
    Ut_fp2 = U_fp2 - U_fp3
    Ut_fp3 = U_fp3 - U_fp4
    Ut_fp4 = U_fp4 - U_fp5
    Ut_fp5 = U_fp5 - U_fp6
    Ut_fp6 = U_fp6 - U_fp7
    Ut_fp7 = U_fp7 - U_fp8
    

    temp1 = (U_fp5+U_fp7)
    temp1_t = (Ut_fp5+Ut_fp7)

    temp2 = (U_fp6*3)
    temp2_t = (Ut_fp6*3)

    temp3 = (U_fp5 - U_fp7)
    temp3_t = (Ut_fp5 - Ut_fp7)

    temp4 = (U_fp5*2)
    temp4_t = (Ut_fp5*2)

    temp5 = (U_fp7)
    temp5_t = (Ut_fp7)

    temp6 = (U_fp5 - U_fp6)
    temp6_t = (Ut_fp5 - Ut_fp6)

    temp7 = (U_fp5 + U_fp6)
    temp7_t = (Ut_fp5 + Ut_fp6)

    temp8 = (U_fp6 + U_fp7)
    temp8_t = (Ut_fp6 + Ut_fp7)

    temp7 = (U_fp5 + U_fp6)
    temp7_t = (Ut_fp5 + Ut_fp6)

    temp8 = (U_fp6 + U_fp7)
    temp8_t = (Ut_fp6 + Ut_fp7)

    temp11 = (U_fp5+U_fp7)
    temp11_t = (Ut_fp5+Ut_fp7)

    temp12 = (U_fp6*3)
    temp12_t = (Ut_fp6*3)

    temp13 = (U_fp5 - U_fp7)
    temp13_t = (Ut_fp5 - Ut_fp7)

    temp14 = (U_fp5*2)
    temp14_t = (Ut_fp5*2)

    temp15 = (U_fp7)
    temp15_t = (Ut_fp7)

    temp16 = (U_fp5 - U_fp6)
    temp16_t = (Ut_fp5 - Ut_fp6)

    temp17 = (U_fp5 + U_fp6)
    temp17_t = (Ut_fp5 + Ut_fp6)

    temp18 = (U_fp6 + U_fp7)
    temp18_t = (Ut_fp6 + Ut_fp7)



    app1 = np.append(temp1,temp2,axis=0)
    app2 = np.append(temp2,temp3,axis=0)
    app3 = np.append(temp4,temp5,axis=0)
    app4 = np.append(temp6,temp7,axis=0)

    app12 = np.append(app1,app2,axis=0)
    app34 = np.append(app3,app4,axis=0)


    app1_t = np.append(temp1,temp2,axis=0)
    app2_t = np.append(temp2,temp3,axis=0)
    app3_t = np.append(temp4,temp5,axis=0)
    app4_t = np.append(temp6,temp7,axis=0)

    app12_t = np.append(app1_t,app2_t,axis=0)
    app34_t = np.append(app3_t,app4_t,axis=0)


    U_fp_append  = np.append(app12,app34,axis=0)
    Ut_fp_append  = np.append(app12_t,app34_t,axis=0)

    return U_fp_append, Ut_fp_append


