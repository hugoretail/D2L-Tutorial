# Multilayer Perceptrons

Resume at: 5.4. Numerical Stability and Initialization

Mistral : https://chat.mistral.ai/chat/e2910dc9-8973-479f-bcfc-49d714b3c703

## 5.1.2 Activation Functions

- ReLU
- pReLU
- Sigmoid
- Tanh

## 5.1.4 Exercises

1. x
2. pReLU(x) = { x si x >= 0
              { ax si x < 0
  dérivée de pRelu(X) = { 1 si x >= 0
                        { a si x < 0
3. Chaque neurone ReLU/pReLU est une fonction linéaire par morceaux ; la compo de fonctions linéaires par morceaux est linéaire aussi ; donc on garde la continuité parce que ReLU/pReLU sont continues.
4. max(tanh(x)) = 1 et max(sigmoid(x)) = 0.25 ; Il suffit de calculer avec :
> tanh(x)=(ex+e−x)/(ex−e−x​),σ(2x)=1/(1+e−2x).
> ...
5. 

