import torch
from hw3_q2 import svm_solver, svm_predictor

def run_basic_test():
    print("===== BASIC LINEAR TEST =====")

    x_train = torch.tensor([[0.,0.],
                            [2.,0.],
                            [1.,1.]])
    y_train = torch.tensor([1.,1.,-1.])

    alpha = svm_solver(x_train, y_train,
                       lr=0.01,
                       num_iters=2000,
                       c=1.0)

    pred = svm_predictor(alpha, x_train, y_train, x_train)

    print("alpha:", alpha)
    print("raw predictions:", pred)
    print("predicted labels:", torch.sign(pred))
    print("true labels:", y_train)
    print("sum alpha_i y_i:", torch.dot(alpha, y_train))
    print()

    assert torch.all(torch.sign(pred) == y_train), "❌ Classification failed"
    assert abs(torch.dot(alpha, y_train)) < 1e-4, "❌ Equality constraint violated"

    print("✔ BASIC TEST PASSED\n")


def run_soft_margin_test():
    print("===== SOFT MARGIN TEST =====")

    x_train = torch.tensor([[0.,0.],
                            [2.,0.],
                            [1.,1.],
                            [0.5,0.5]])
    y_train = torch.tensor([1.,1.,-1.,1.])

    alpha = svm_solver(x_train, y_train,
                       lr=0.01,
                       num_iters=3000,
                       c=1.0)

    pred = svm_predictor(alpha, x_train, y_train, x_train)

    print("alpha:", alpha)
    print("raw predictions:", pred)
    print("predicted labels:", torch.sign(pred))
    print("true labels:", y_train)
    print("sum alpha_i y_i:", torch.dot(alpha, y_train))
    print()

    assert abs(torch.dot(alpha, y_train)) < 1e-4, "❌ Equality constraint violated"

    print("✔ SOFT MARGIN TEST PASSED\n")


def run_large_split_test():
    print("===== 1000 TRAIN / 500 TEST =====")

    torch.manual_seed(42)

    N_train = 1000
    N_test = 500
    D = 10
    
    X = torch.randn(N_train + N_test, D)
    w_true = torch.randn(D)

    y = torch.sign(X @ w_true)
    y[y == 0] = 1

    x_train = X[:N_train]
    y_train = y[:N_train]

    x_test = X[N_train:]
    y_test = y[N_train:]

    
    alpha = svm_solver(
        x_train, y_train,
        lr=0.001,
        num_iters=300,
        c=1.0
    )

    raw_pred = svm_predictor(alpha, x_train, y_train, x_test)
    pred = torch.sign(raw_pred)

    acc = (pred == y_test).float().mean()

    print("Test Accuracy:", acc.item())
    print("sum alpha_i y_i:", torch.dot(alpha, y_train).item())
    print("Support vectors:", torch.sum(alpha > 1e-6).item())

    assert acc > 0.9
    assert abs(torch.dot(alpha, y_train)) < 1e-3

    print("✔ LARGE SPLIT TEST PASSED\n")


if __name__ == "__main__":
    run_basic_test()
    run_soft_margin_test()
    run_large_split_test()
    print("🎉 ALL TESTS PASSED")