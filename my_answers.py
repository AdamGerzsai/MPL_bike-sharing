import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))


    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):        
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # TODO:
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # TODO:
        final_outputs = final_inputs        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = y-final_outputs
        output_error_term = error[:,None]         
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)        
#         print(output_error_term)
#         print(self.weights_hidden_to_output)
#         print(self.weights_hidden_to_output.T)
#         print(hidden_error)
#         print(hidden_error_term)

#         print(output_error_term,' ',output_error_term.shape)
#         print(hidden_outputs,' ',hidden_outputs.shape)
        
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        
#         print(' ')
#         print(delta_weights_i_h)
#         print(delta_weights_h_o)
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.8
hidden_nodes = 32
output_nodes = 1

# 1000
# 1.5 3 = 0.161
# 2.0 3 = 0.119

# 1500
# 2.5 3 0,08
# 3.0 4 0.074
# 2.0 5 0.07

# ./first-neural-network/setup.sh

# ./first-neural-network/setup.sh

# iterations = 100
# learning_rate = 0.1
# hidden_nodes = 64
# output_nodes = 1