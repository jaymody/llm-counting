from sympy import Function, MatrixSymbol, Symbol


def repeat(d, n):
    return {k: repeat(v, n) if isinstance(v, dict) else n * v for k, v in d.items()}


def count(d):
    return sum([count(v) if isinstance(v, dict) else v for v in d.values()])


def tree_map(d, f):
    return {k: tree_map(v, f) if isinstance(v, dict) else f(v) for k, v in d.items()}


class Layer:
    @property
    def flops_dict(self):
        raise NotImplementedError()

    @property
    def flops_count(self):
        return count(self.flops_dict)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()


class Embeddings(Layer):
    def __init__(self, l, d_model, n_vocab):
        super().__init__()
        self.l, self.d_model, self.n_vocab = l, d_model, n_vocab

    def evaluate(self):
        X_word = MatrixSymbol(r"X_{word}", self.l, self.d_model)
        X_pos = MatrixSymbol(r"X_{pos}", self.l, self.d_model)

        return X_word + X_pos

    @property
    def flops_dict(self):
        return {
            "X_word + X_pos": self.l * self.d_model,
        }


class Softmax(Layer):
    def __init__(self, m, n):
        super().__init__()
        self.m, self.n = m, n

    @property
    def flops_dict(self):
        return {
            "x - max(x)": self.m * self.n,
            "e^x": self.m * self.n,
            "sum": self.m * self.n,
        }


class ScaledDotProductAttention(Layer):
    def __init__(self, n_q, n_k, d_k, d_v):
        super().__init__()
        self.n_q, self.n_k, self.d_k, self.d_v = n_q, n_k, d_k, d_v

    def evaluate(self):
        # Q = MatrixSymbol("Q", self.n_q, self.d_k)
        # K = MatrixSymbol("K", self.n_k, self.d_k)
        # V = MatrixSymbol("Q", self.n_k, self.d_v)
        # mask = MatrixSymbol("Q", self.n_q, self.n_k)
        pass

    @property
    def flops_dict(self):
        return {
            "Q * K.T": 2 * self.n_q * self.d_k * self.n_k,
            "/ sqrt(d_k)": self.n_q * self.n_k,
            "+ mask": self.n_q * self.n_k,
            "Softmax": Softmax(self.n_q, self.n_k).flops_dict,
            "* V": 2 * self.n_q * self.n_k * self.d_v,
        }


class MultiHeadAttention(Layer):
    def __init__(self, n_q, n_k, d_k, d_v, d_model, n_heads):
        super().__init__()
        self.n_q = n_q
        self.n_k = n_k
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

    def evaluate(self):
        # WQ = MatrixSymbol("WQ", d_model, d_k * n_heads)
        # WK = MatrixSymbol("WK", d_model, d_k * n_heads)
        # WV = MatrixSymbol("WV", d_model, d_v * n_heads)
        # WO = MatrixSymbol("WO", d_v * n_heads, d_model)
        pass

    @property
    def flops_dict(self):
        # the FLOPs count is the same no matter the number of heads, the heads just
        # enable different representation subspaces, it doesn't add any computational
        # complexity as far as FLOPs is concerned
        return {
            "Q * WQ": 2 * self.n_q * self.d_model * (self.d_k * self.n_heads),
            "K * WK": 2 * self.n_k * self.d_model * (self.d_k * self.n_heads),
            "V * WV": 2 * self.n_k * self.d_model * (self.d_v * self.n_heads),
            "ScaledDotProductAttention": repeat(
                ScaledDotProductAttention(
                    self.n_q, self.n_k, self.d_k, self.d_v
                ).flops_dict,
                self.n_heads,
            ),
            "* WO": 2 * self.n_q * (self.d_v * self.n_heads) * self.d_model,
        }


class PositionWiseFFN(Layer):
    def __init__(self, l, d_in, d_ff, d_out):
        super().__init__()
        self.l, self.d_in, self.d_ff, self.d_out = l, d_in, d_ff, d_out

    def evaluate(self):
        # W1 = MatrixSymbol("W1", d_in, d_ff)
        # b1 = MatrixSymbol("b1", d_ff, 1)

        # W1 = MatrixSymbol("W1", d_ff, d_out)
        # b1 = MatrixSymbol("b1", d_out, 1)

        # X = MatrixSymbol("X", l, d_in)
        # output = relu(X * W1 + b1) * W2 + b2
        pass

    @property
    def flops_dict(self):
        return {
            "* W1": 2 * self.l * self.d_in * self.d_ff,
            "+ b1": self.d_ff,
            "* W2": 2 * self.l * self.d_ff * self.d_out,
            "+ b2": self.d_out,
        }


class LayerNorm(Layer):
    def __init__(self, m, n):
        super().__init__()
        self.m, self.n = m, n

    def evaluate(self):
        # X = MatrixSymbol("X", n, m)
        #
        # mean_x = sum(X, axis=-1) / m
        # var_x = sum(X**2, axis=-1) / m - mean_x**2
        #
        # numer = x - mean_x
        # denom = sqrt(var_x + eps)
        #
        # gamma * (numer / denom) + beta
        pass

    @property
    def flops_dict(self):
        n, m = self.n, self.m
        return {
            "mean(x)": m * n
            + n,  # element wise add and then a divide over the resulting vector
            "var_x": {
                "X**2": m * n,
                "sum": m * n,
                "/ m": n,
                "mean_x**2": n,
                "m - mean_x**2": n,
            },
            "numerator": n,  # x - mean_x
            "denominator": 2
            * n,  # sqrt(var_x + eps), elem wise addition then elem wise sqrt
            "gamma*(numerator/denominator)+beta": 3
            * m
            * n,  # after broadcasting, 1 elem wise divide, multiply, then add
        }


class Block(Layer):
    def __init__(self, l, d_model, d_ff, n_heads):
        super().__init__()
        self.l, self.d_model, self.d_ff, self.n_heads = l, d_model, d_ff, n_heads

    def evaluate(self):
        # X = MatrixSymbol("X", l, d_model)
        # X = X + MultiHeadAttention(LayerNorm(X))
        # X = X + PositionWiseFFN(LayerNorm(X))
        pass

    @property
    def flops_dict(self):
        return {
            "LayerNorms": repeat(LayerNorm(self.l, self.d_model).flops_dict, 2),
            "ResidualConnections": 2 * self.l * self.d_model,
            "MultiHeadAttention": MultiHeadAttention(
                self.l,
                self.l,
                self.d_model / self.n_heads,
                self.d_model / self.n_heads,
                self.d_model,
                self.n_heads,
            ).flops_dict,
            "PositionWiseFFN": PositionWiseFFN(
                self.l, self.d_model, self.d_ff, self.d_model
            ).flops_dict,
        }


class Transformer(Layer):
    def __init__(self, l, d_model, d_ff, n_vocab, n_layers, n_heads):
        super().__init__()
        self.l, self.d_model, self.d_ff, self.n_vocab, self.n_layers, self.n_heads = (
            l,
            d_model,
            d_ff,
            n_vocab,
            n_layers,
            n_heads,
        )

    def evaluate(self):
        # X = embeddings
        # for _ in range(n_layers):
        #   X = block(X)
        # X = layer_norm(X)
        # W_lm = MatrixSymbol("LM", d_model, n_vocab) # projection_to_vocab (lm head)
        # outputs = X * W_lm
        pass

    @property
    def flops_dict(self):
        return {
            "Embeddings": Embeddings(self.l, self.d_model, self.n_vocab).flops_dict,
            "LayerNorm_Final": LayerNorm(self.l, self.d_model).flops_dict,
            "ProjectToVocab": 2 * self.l * self.d_model * self.n_vocab,
            "Blocks": repeat(
                Block(self.l, self.d_model, self.d_ff, self.n_heads).flops_dict,
                self.n_layers,
            ),
        }
