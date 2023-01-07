import numpy as np
import random


Function = input("Activation Functions:\nLeakyRelu (I have found this to be much better) (LR)\nSigmoid (S)\n")

#Leaky RELU and Sigmoid Activation Functions
def ActivationFunction(x):
    if Function.lower() == "lr":
        if x > 0:
            return x
        else: 
            return 0.1*x
    elif Function.lower() == "s":
        return 1/(1 + np.exp(-x))
    
#Leaky RELU and Sigmoid Derivative Functions
def ActivationFunctionP(x):
    if Function.lower() == "lr":
        if x > 0 :
            return 1
        else: 
            return 0.1
    elif Function.lower() == "s":
        return ActivationFunction(x)*(1-ActivationFunction(x))

#Defining attributes of a Node:
class node: 
    id = int()#Identifier assigned to all nodes 
    layer = int()
    PreviousNodes = int()#Amount of Nodes in the previous layer 
    IsInput = bool()#Is the Node in the input layer?
    PrevNodeId = []#Identifiers of all the nodes in the previous layer
    PrevWeightVal = []#Weights of all the nodes in the previous layer
    bias = float()
    a = float()#Value of the node after the activation function
    z = float()#Value of the node before activation function

#Function for creating the nodes based on the desired architecture 
def createNodes(_architecture):
    
    global total, input, layers, architecture
    architecture = _architecture#List of the amount of nodes in each layer 
    layers = len(architecture)#number of layers in the network
    input = architecture[0]#Amount of nodes in the input layer
    total = 0
    for i in architecture: total+=i#Total number of nodes in the network


    NodeId = 0
    """
    This loop creates all the nodes required in the architecture of the network
    and gives them all their respective attributes 
    """
    for index,layer in enumerate(architecture):
        for i in range(layer):
            newNode = node()
            newNode.id = NodeId#Every node has a unique identifier in order to allow me to change attributes of a specific node
            newNode.z = 0#The value of the node before it goes through the Activation Function
            newNode.a = 0#The value of the node after it goes through the Activation Function 
            newNode.bias = 0
            newNode.layer = index#What layer the node is in
            if index == 0:
                newNode.PreviousNodes = 0#If the node is in the input layer it will have no Nodes before it
            else:
                newNode.PreviousNodes = architecture[index-1]
                
                """
                Identifies how many nodes are in the previous layer↓↓
                """
                top = 0
                for i in range(index-1, -1, -1):
                    top += architecture[i]
                bottom = top-architecture[index-1]
                newNode.PrevNodeId = [i for i in range(bottom, top)]
            
            
            newNode.PrevWeightVal = [random.random()-0.5 for i in range(architecture[index-1])]#Weight values are initialized to be a random number between -0.5 and 0.5 
            Nodes.append(newNode)#All nodes that are created are appended to a list that will contain every node in the network. The ID's of the nodes are equal to the index they have in the list
    
    """
    Layer Start is a list of node ID's that represent the beginning of each layer
    """
    global LayerStart
    LayerStart = [] 
    PrevLayer = -1
    for i in range(len(Nodes)):
        if Nodes[i].layer != PrevLayer:
            LayerStart.append(i)
            PrevLayer = Nodes[i].layer

"""
When the input data is fed into the input nodes, the values of all the rest of the nodes in the network will change. The calculation of these new values
is done through Forward Propagation 
"""
def ForwardPropagate():
    for i in range(input):
        Nodes[i].a = ActivationFunction(Nodes[i].z)
    for i in range(input, total):
        Nodes[i].z = Nodes[i].bias
        for y in range(Nodes[i].PreviousNodes):

            Nodes[i].z += Nodes[Nodes[i].PrevNodeId[y]].a*Nodes[i].PrevWeightVal[y]
        Nodes[i].a = ActivationFunction(Nodes[i].z)

"""
The input data needs to be shuffled at the beginning of every epoch in order to prevent any bias and stop the network
from just learning the order of the training.
"""
def Shuffle(InData, OutData):
    newlist = []
    for i in range(len(InData)):
        newlist.append(i)
    random.shuffle(newlist)
    InData2 = InData
    OutData2 = OutData
    for index, i in enumerate(newlist):
        InData[index] = InData2[i]
        OutData[index] = OutData2[i]


"""
It is important that each data point in the input data (InData variable) has an index that corresponds to the label in the OutData variable
"""
OutData = []#Labels for the dataset
InData = np.loadtxt("C:\\Users\\TeenT\\OneDrive\\Desktop\\Neural Network\\mnist_train.csv", delimiter=",")#The dataset that will be inputed into the network


##################################################
"""
This is all just manipulation of the dataset
"""
for data in InData:
    OutData.append([int(data[0])])

for i in OutData:
    content = i[0]
    i.pop(0)
    for x in range(content):
        i.append(0)
    i.append(1)
    for x in range(9-content):
        i.append(0)
InData = InData.tolist()

for index, i in enumerate(InData):
     InData[index] = i[1:]
     


for index, data_point in enumerate(InData):
    for index2, data in enumerate(data_point):
        InData[index][index2] /= 255
        

InData = InData[0:10000]#The mnist dataset is quite long so here i can decide how much of the dataset i want to use. Eg: 1000 out of the 60000

##################################################


#The main Train function↓
def Train(InData, OutData, Batches, Epochs, LR):
    cost = 0#Cost is initialized to 0
    
    WeightDerivativeList = []#List of all the weight derivatives
    BiasDerivativeList = []#List of all the bias derivatives
    baseDerivatives = []#List of the base derivatives 
    
    #Initialization of the lists above
    for i in range(len(Nodes)):
        BiasDerivativeList.append(0)
        baseDerivatives.append(0)
        WeightDerivativeList.append([])
    for i in range(input, len(Nodes)):
        for j in range(Nodes[i].PreviousNodes):
            WeightDerivativeList[i].append(0)
            
    #Main loop for the training
    for Epoch in range(Epochs):
        cost = 0#Cost is reset to 0
        Shuffle(InData, OutData)#Data is shuffled 
        
        for Data_index in range(len(InData)):    
            #Input data is fed into the network ↓
            for i in range(len(InData[Data_index])):
                Nodes[i].z = InData[Data_index][i]
            
            #Forward Prpagate after the inputs are fed in ↓
            ForwardPropagate()
            
            #Cost Calculation
            for i in range(architecture[-1]):
                cost += ((OutData[Data_index][i]-Nodes[LayerStart[-1]+i].a))**2
            
            #####################################################################################
            """
            Base derivatives are calculated and used to find the weight and bias derivatives
            """
            for NODE in range(len(Nodes)-1, -1 ,-1):
                if Nodes[NODE].layer == layers-1:
                    baseDerivatives[NODE]= -2*(OutData[Data_index][NODE-LayerStart[layers-1]]-Nodes[NODE].a)*(ActivationFunctionP(Nodes[NODE].z))
                else:
                    baseDerivatives[NODE] = 0
                    for n in range(architecture[Nodes[NODE].layer+1]):
                        baseDerivatives[NODE] += baseDerivatives[LayerStart[Nodes[NODE].layer+1]+n]*Nodes[LayerStart[Nodes[NODE].layer+1]+n].PrevWeightVal[NODE-LayerStart[Nodes[NODE].layer]]
                    baseDerivatives[NODE] *= ActivationFunctionP(Nodes[NODE].z)
                    
                BiasDerivativeList[NODE] += baseDerivatives[NODE]
                if Nodes[NODE].layer != 0:
                    for i in range(Nodes[NODE].PreviousNodes):
                        WeightDerivativeList[NODE][i] += baseDerivatives[NODE]*Nodes[LayerStart[Nodes[NODE].layer-1] +i].a#
                
            if Data_index%Batches==0 and Data_index!=0:
                for NODE in range(len(Nodes)):
                    Nodes[NODE].bias -= BiasDerivativeList[NODE] * LR / Batches
                    BiasDerivativeList[NODE] = 0
                    
                    for i in range(Nodes[NODE].PreviousNodes):
                        Nodes[NODE].PrevWeightVal[i] -= WeightDerivativeList[NODE][i] * LR / Batches
                        WeightDerivativeList[NODE][i] = 0
             #####################################################################################
        
        #finding the average cost
        cost /= len(InData)
        
        cost = '{0:.30f}'.format(cost)
        print(f"Cost: {cost} (Epoch {Epoch})")

#List of all the nodes in the network        
Nodes = []
"""
The architecture of the network: 784 input nodes(one for each pixel in the 28x28 image), 30 nodes in the 2nd layer (Hidden Layer),
30 nodes in the 3rd layer (Hidden Layer) and 10 nodes in the output layer (representing every possible output. Ie: 0,1,2,3,4,5,6,7,8,9)
"""
createNodes([784,30,30,10])
#Calling the Train function to run 40 epochs with a 32 batch size and a 0.01 learning rate
Train(InData, OutData, 32, 40, 0.01)


#This is a predict function that will return the value of the output nodes when test data is inputed
def predict(data):
    for i in range(input):
        Nodes[i].z = data[i]
    ForwardPropagate()
    
    OutputNodeValues = []
    for i in range(10):
        OutputNodeValues.append(Nodes[LayerStart[-1]+i].z)
    return OutputNodeValues

########################################################################################
"""
Manipulation of the testing data
"""
TestOutData = []
TestInData = np.loadtxt("C:\\Users\\TeenT\\OneDrive\\Desktop\\Neural Network\\mnist_test .csv", delimiter=",")

for data in TestInData:
    TestOutData.append([int(data[0])])

for i in OutData:
    content = i[0]
    i.pop(0)
    for x in range(content):
        i.append(0)
    i.append(1)
    for x in range(9-content):
        i.append(0)

TestInData = TestInData.tolist()

for index, i in enumerate(TestInData):
     TestInData[index] = i[1:]
     
for index, data_point in enumerate(TestInData):
    for index2, data in enumerate(data_point):
        TestInData[index][index2] /= 255

TestInData = TestInData[0:100]
#########################################################################################


for index, i in enumerate(TestInData):
    response = predict(i) 
    print(f"\nThe picture was a: {TestOutData[index][0]}")
    x = 0
    y = 0
    for index, i in enumerate(response):
        if i > x:
            x = i
            y = index
    print(f"The prediction was: {y}")
