using QuadGK

integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
