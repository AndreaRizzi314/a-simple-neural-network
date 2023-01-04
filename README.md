
# Table of contents
- [Table of contents](#table-of-contents)
- [Overview of the project](#overview-of-the-project)
  - [Architecture](#architecture)
  - [Calculating Biases and Weights](#calculating-biases-and-weights)
    - [Hidden Layer --> Output Layer weight derivatives](#hidden-layer----output-layer-weight-derivatives)
    - [Input Layer --> Hidden Layer weight derivatives](#input-layer----hidden-layer-weight-derivatives)
    - [Output Layer bias derivatives](#output-layer-bias-derivatives)
    - [Hidden Layer bias derivatives](#hidden-layer-bias-derivatives)
    - [Input Layer bias derivatives](#input-layer-bias-derivatives)
  - [Prerequisites](#prerequisites)
    - [Modules](#modules)
  - [Things I'm still working on](#things-im-still-working-on)


# Overview of the project    
This is my investigation into Neural Networks and the math behind them. This neural network is designed to learn the mnist dataset and be able to recognise handwritten digits



## Architecture 

## Calculating Biases and Weights
### Hidden Layer --> Output Layer weight derivatives
![image](https://user-images.githubusercontent.com/80152624/199080179-a7eb1aee-37cf-42f7-a86d-e152ff4550ae.png)
### Input Layer --> Hidden Layer weight derivatives
![image](https://user-images.githubusercontent.com/80152624/199084001-873fdfd1-bd4a-4855-a7de-2617e730d1f3.png)
### Output Layer bias derivatives
![image](https://user-images.githubusercontent.com/80152624/199084679-c498d4a2-1790-4ba8-8363-d6d813e13ae8.png)
### Hidden Layer bias derivatives
![image](https://user-images.githubusercontent.com/80152624/199085280-8f45c141-1d60-44f8-8c5d-1401494ac6b1.png)
### Input Layer bias derivatives
![image](https://user-images.githubusercontent.com/80152624/199086279-7e337fec-c83c-404a-ac99-b96ed4dfb370.png)



## Prerequisites 
### Modules
There is one module used in this program that does not come as default in python:\
\
**Numpy:**
```
pip install numpy
```
## Things I'm still working on
Although my math is correct for the calculation of biases, I'm struggling to implement them without the network becoming very unreliable. Until this problem is solved, I will only be using weights to train the network. As a result I am restricted to the approximation of linear functions.


