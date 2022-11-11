import numpy as np
from typing import Optional, Tuple, Union


class Dense:
    def init(self, n_in: int, n_out: int, use_bias: bool = True):
        self.use_bias_ = use_bias

        self.weights_ = np.random.random((n_in, n_out))
        self.input_ = None

        if self.use_bias_:
            self.intercept_ = np.random.random(n_out)

    @property
    def weights(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray]]:
        if self.use_bias_:
            return self.weights_, self.intercept_

        return (self.weights_, None)

    @property
    def input(self) -> np.ndarray:
        return self.input_

    def call(self, x: np.ndarray) -> np.ndarray:
        self.input_ = x.copy()

        if self.use_bias_:
            return x @ self.weights_ + self.intercept_

        return x @ self.weights_

    def grad(self, gradOutput: np.ndarray) -> Union[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, Tuple[np.ndarray]]]:
        grad_input = gradOutput @ self.weights_
        grad_W = gradOutput.T @ self.input_

        if self.use_bias_:
            grad_b = np.sum(gradOutput, axis=0)
            return (grad_input, (grad_W, grad_b))

        return (grad_input, (grad_W, None))
