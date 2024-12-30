import tensorflow as tf
from keras.src.optimizers import optimizer


class AdaGradOptimizer(optimizer.Optimizer):
    def __init__(
        self, learning_rate=0.01, epsilon=1e-7, name="AdaGradFromScratch", **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.epsilon = epsilon

    def build(self, var_list):
        if self.built:
            return
        # Create slots to store the sum of squared gradients for each variable
        super().build(var_list)
        self._accumulators = [
            self.add_variable(shape=var.shape, name="accumulator", initializer="zeros")
            for var in var_list
        ]

    def update_step(self, gradient, variable, learning_rate):
        # Get the accumulator for this variable
        accumulator = self._accumulators[self._get_variable_index(variable)]
        # Update the accumulator with the square of the gradient
        accumulator.assign_add(tf.square(gradient))
        # Update the variable using the adjusted gradient
        adjusted_gradient = gradient / (tf.sqrt(accumulator + self.epsilon))
        variable.assign_sub(learning_rate * adjusted_gradient)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
            }
        )
        return config
