# AI Crystal Structure Classification

This project uses artificial intelligence to classify crystal structures on an arbitrary scale. It started as a personal learning endeavor after attending several ML workshops hosted by the society. Inspired by an internship focused on classifying nano-crystal structures, I decided to build a mini version of the project to deepen my understanding of both crystal classification and the underlying data acquisition methods. I used a neural network to make a multiclass classification machine learning model. First, I made a neural network to find patterns and trends in the data using ReLU. It is trained by using sparse categorical cross-entropy, which is just a fancy way of saying measuring how wrong a guess from the neural network is by comparing it to the real answer. Then, to make the model faster, an Adam optimizer is used, which is just saying adjusting the settings/weights, which are the connections between neurons, to improve the predictions. In the hidden layers of the neural network, the ReLU decides whether the data in the neural network is important and isn't a random prediction. Then, in the output layer, it produces a probability based on the neural network's predictions into percentages and chooses the one with the highest probability as the answer. Now you can just change what you want to be predicted using unseen data you give to the model. Note that the main challenge of this project is the inability to obtain real experimental data, so I had to create a synthetic dataset.
VERY GOOD IDEA TO JUST RUN IT IN GITHUB CODEBASES IT IS JUST MORE SIMPLE, Also just ignore the cpu/gpu errors its not important.

## How to Run

1. **Install Dependencies:**

   ```bash
   pip install numpy pandas scikit-learn matplotlib tensorflow

2.Run main.py

let me know if there can be any improvents, just want to learn :)


