def demo_linearized_gpr():

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # matplotlib.use("Agg")

    rng = np.random.RandomState(0)

    # Generate sample data
    noise = 1.0
    input_noise = 0.2
    n_train = 1_000
    n_test = 1_000
    n_inducing = 100
    batch_size = None
    X = 15 * rng.rand(n_train, 1)

    def plot_results(title=None):
        # Plot results
        plt.figure(figsize=(10, 5))
        lw = 2
        plt.scatter(X, y, c="k", label="data")
        plt.plot(X_plot, np.sin(X_plot), color="navy", lw=lw, label="True")

        plt.plot(X_plot, y_gpr, color="darkorange", lw=lw, label="GPR")
        plt.fill_between(
            X_plot[:, 0],
            (y_gpr - 2 * y_std).squeeze(),
            (y_gpr + 2 * y_std).squeeze(),
            color="darkorange",
            alpha=0.2,
        )
        plt.xlabel("data")
        plt.ylabel("target")
        plt.xlim(0, 20)
        plt.ylim(-4, 4)
        if title is not None:
            plt.title(title)
        plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
        plt.show()

    def f(x):
        return np.sin(x)

    y = f(X)

    X += input_noise * rng.randn(X.shape[0], X.shape[1])
    y += noise * (0.5 - rng.rand(X.shape[0], X.shape[1]))  # add noise
    X_plot = np.linspace(0, 20, n_test)[:, None]
    X_plot += input_noise * rng.randn(X_plot.shape[0], X_plot.shape[1])
    X_plot = np.sort(X_plot, axis=0)

    X_variance = input_noise
    n_restarts = 0
    verbose = 1
    normalize_y = False
    max_iters = 500

    # ==================================
    # Standard GPR
    # ==================================
    gpr_clf = GPRegressor(
        verbose=verbose,
        n_restarts=n_restarts,
        X_variance=X_variance,
        normalize_y=normalize_y,
    )

    gpr_clf.fit(X, y)

    y_gpr, y_std = gpr_clf.predict(
        X_plot, return_std=True, noiseless=False, linearized=False
    )
    print(gpr_clf.display_model())
    plot_results("GPR")

    y_gpr, y_std = gpr_clf.predict(
        X_plot, return_std=True, noiseless=False, linearized=True
    )
    print(gpr_clf.display_model())
    plot_results("GPR")

    # ==================================
    # Sparse GPR
    # ==================================
    gpr_clf = SparseGPRegressor(
        verbose=verbose,
        n_restarts=n_restarts,
        X_variance=X_variance,
        normalize_y=normalize_y,
        max_iters=max_iters,
        n_inducing=n_inducing,
    )

    gpr_clf.fit(X, y)

    y_gpr, y_std = gpr_clf.predict(
        X_plot, return_std=True, noiseless=False, linearized=False
    )
    print(gpr_clf.display_model())
    plot_results("SGPR")

    y_gpr, y_std = gpr_clf.predict(
        X_plot, return_std=True, noiseless=False, linearized=True
    )
    print(gpr_clf.display_model())
    plot_results("SGPR")

    # ==================================
    # Sparse GPR
    # ==================================
    gpr_clf = UncertainSGPRegressor(
        verbose=verbose,
        n_restarts=n_restarts,
        X_variance=X_variance,
        normalize_y=normalize_y,
        max_iters=max_iters,
        n_inducing=n_inducing,
        batch_size=batch_size,
    )

    gpr_clf.fit(X, y)

    y_gpr, y_std = gpr_clf.predict(
        X_plot, return_std=True, noiseless=False, linearized=True
    )
    print(gpr_clf.display_model())
    plot_results("SVGPR")

    y_gpr, y_std = gpr_clf.predict(X_plot, return_std=True, noiseless=False,)
    print(gpr_clf.display_model())
    plot_results("SVGPR")

    return None


if __name__ == "__main__":
    demo_linearized_gpr()
