import numpy as np
import random

seperate = "-------------------------------------------\n"
Function = input("Activation Functions:\nLeakyRelu (LR)\nSigmoid (S)\n")

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
    
    
#Function for creating the nodes based on the desired architechture 
def createNodes(architecture):
    global input
    input = architecture[0]
    HL = architecture[1]
    output = architecture[2]
    global total
    total = input+output+HL

    for i in range(total):
        newNode = node()
        newNode.id = i
        newNode.z = 0
        newNode.a = 0.5
        newNode.bias = random.random()*2-1
        if i < input:
            newNode.IsInput = True
            newNode.layer = 0
            newNode.PreviousNodes = 0
            newNode.PrevNodeId = []
            newNode.PrevWeightVal = []
        elif i < HL+input:
            newNode.IsInput = False
            newNode.layer = 1
            newNode.PreviousNodes = input
            newNode.PrevNodeId = []
            newNode.PrevWeightVal = []
            for i in range(input):
                newNode.PrevNodeId.append(i)
                newNode.PrevWeightVal.append((random.random()*2)-1)
        else:
            newNode.IsInput = False
            newNode.layer = 2
            newNode.PreviousNodes = HL
            newNode.PrevNodeId = []
            newNode.PrevWeightVal = []
            for i in range(HL):
                newNode.PrevNodeId.append(i+input)
                newNode.PrevWeightVal.append((random.random()*2)-1)
        Nodes.append(newNode)
        


def createInputList(list, length, botLim, topLim):
    while True:
       
        x = random.random()*(topLim-botLim) + botLim
        
        if not x in list:
            list.append(x)
        if len(list) == length:
            break

def ForwardPropagate():
    for i in range(1,total):
        Nodes[i].z = 0#Nodes[i].bias
        for y in range(Nodes[i].PreviousNodes):
            Nodes[i].z += Nodes[Nodes[i].PrevNodeId[y]].a*Nodes[i].PrevWeightVal[y]
            Nodes[i].a = ActivationFunction(Nodes[i].z)

#Calculating Output Derivative
def OutputDerivative(Y0):
    for i in Nodes[len(Nodes)-1].PrevNodeId:
        
        derivative = -2*(Y0-Nodes[len(Nodes)-1].a)*(ActivationFunctionP(Nodes[len(Nodes)-1].z))*(Nodes[i].a)
        WeightDerivativeList[len(Nodes)-1][i-Nodes[len(Nodes)-1].PrevNodeId[0]] += derivative 
             
    #Calculating Bias Derivatives
    derivative = -2*(Y0-Nodes[len(Nodes)-1].a)*(ActivationFunctionP(Nodes[len(Nodes)-1].z))
    BiasDerivativeList[i] += derivative
    
#Calculating Hidden Layer Derivatives    
def HiddenLayerDerivative(Y0):
    for i in Nodes[len(Nodes)-1].PrevNodeId:
        derivative = -2*(Y0-Nodes[len(Nodes)-1].a)*(ActivationFunctionP(Nodes[len(Nodes)-1].z))*Nodes[len(Nodes)-1].PrevWeightVal[i-Nodes[len(Nodes)-1].PrevNodeId[0]]*(ActivationFunctionP(Nodes[i].z))*(Nodes[0].a)
        WeightDerivativeList[i-Nodes[len(Nodes)-1].PrevNodeId[0]+1][0] += derivative
        
        #Calculating Bias Derivatives
        derivative = -2*(Y0-Nodes[len(Nodes)-1].a)*(ActivationFunctionP(Nodes[len(Nodes)-1].z))*Nodes[len(Nodes)-1].PrevWeightVal[i-Nodes[len(Nodes)-1].PrevNodeId[0]]*(ActivationFunctionP(Nodes[i].z))
        BiasDerivativeList[i] += derivative
        
#Calculating Input Derivatives(only bias derivative needed)
def InputDerivative(Y0):
    Sum = 0
    for i in Nodes[len(Nodes)-1].PrevNodeId:
        
        Sum += Nodes[len(Nodes)-1].PrevWeightVal[i-Nodes[len(Nodes)-1].PrevNodeId[0]]*(ActivationFunctionP(Nodes[i].z))*Nodes[i].PrevWeightVal[0]

    derivative = -2*(Y0-Nodes[len(Nodes)-1].a)*(ActivationFunctionP(Nodes[len(Nodes)-1].z))*Sum*(ActivationFunctionP(Nodes[0].z))
    BiasDerivativeList[i] += derivative
    
    
Nodes = []#List of all Node classes 
createNodes([1,5,1])#Creating a 1,n,1 architecture for the network.ie(1 input node, n hidden layer nodes, 1 output node)

inputData = []
createInputList(inputData, 1000, 0, 2)#create an input list of length 1000 & numbers between 0 and 2 

print(inputData)
LR = .1#Learning Rate


#Keeps Training until the cost is below 0.0000000000000000000000000001
Cost = 1
Epochs = 0
while Cost >= 0.0000000000000000000000000001: # <-- Epochs

    Epochs += 1
    WeightDerivativeList = []
    BiasDerivativeList = []
    Cost = 0
    
    for j in range(total):
        BiasDerivativeList.append(0)
        
    for k in range(total):
        WeightDerivativeList.append([])
        if k > 0:
            for j in Nodes[k].PrevNodeId:
                WeightDerivativeList[k].append(0)
    
    for value in inputData:
        expectedOutput = ActivationFunction(value*10)#What the network should output. In this case I want it to output a value 3 times greater than the input. ie (Input = 10 : Output = 30)
        
        Nodes[0].z = value#+Nodes[0].bias #input value
        Nodes[0].a = ActivationFunction(Nodes[0].z)
        ForwardPropagate()
        OutputDerivative(expectedOutput)#calculating the output derivatives 
        HiddenLayerDerivative(expectedOutput)#calculating the hidden layer derivatives
        #InputDerivative(expectedOutput)
        Cost += (expectedOutput-Nodes[len(Nodes)-1].a)**2 #
        """
        Cost = ((Value we wanted)-(value we got))^2
        This is used as a measure of how accurate the network is 
        """  
    Cost /= len(inputData)#Getting the average
    Cost = '{0:.33f}'.format(Cost)
    print(f"Cost = {Cost} (Epoch {Epochs+1})")
    
    Cost = float(Cost)
    
    #Changing the weight values
    #Weight-(Learning Rate*Derivative)  = NewWeight 
    for index, derivatives in enumerate(WeightDerivativeList):
        for index2, weight in enumerate(derivatives):
            if index == 0:
                pass
            Nodes[int(index)].PrevWeightVal[index2] -= (LR*weight/len(inputData))
    for index, derivatives in enumerate(BiasDerivativeList):
        Nodes[int(index)].bias -= (LR*derivatives/len(inputData))
        
    for l in range(len(BiasDerivativeList)):
        BiasDerivativeList[l] = 0
    for l in range(len(WeightDerivativeList)):
        if l == 0:
            pass
        for p in range(len(WeightDerivativeList[l])):
            WeightDerivativeList[l][p] = 0
            
    
#Testing a different set of data
TestIn = []
createInputList(TestIn, 10, 0, 2)
for i in TestIn:
    Nodes[0].z = i
    Nodes[0].a = ActivationFunction(Nodes[0].z)
    ForwardPropagate()
    

    print(f"{seperate}",end="")
    print(f"Test input : {Nodes[0].z}")
    print(f"Output: {Nodes[len(Nodes)-1].z}")
