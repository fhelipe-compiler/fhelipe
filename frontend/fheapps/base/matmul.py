import tensorfhe as tfhe
from tensorfhe import Tensor

def mul_mm(a: Tensor, b: Tensor) -> Tensor:
    m, n = a.shape
    _, p = b.shape

    a_rep = a.replicate(dim=2, n=p)
    b_rep = b.replicate(dim=0, n=m)

    partial_products = a_rep * b_rep
    return partial_products.sum(dim=1)

class Matmul(tfhe.App):
    def __init__(self, n: int = 256, **kwargs):
        a = tfhe.tensor("a", (n, n))
        b = tfhe.tensor("b", (n, n))
        result = mul_mm(a, b)
        super().__init__(id=(n,), out=result)

if __name__ == "__main__":
    Matmul_Ct_Ct.main()
