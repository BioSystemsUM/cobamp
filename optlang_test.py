if __name__ == '__main__':

    from pathway_analysis import IrreversibleLinearSystem, DualLinearSystem, KShortestEnumerator
    import numpy as np
    import pandas as pd

    S = np.array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
                  [0, 1, -1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, -1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, -1]])

    irrev = [0, 1, 2, 4, 5, 6, 7, 8]
    T = np.array([0] * S.shape[1]).reshape(1, S.shape[1])
    T[0, 8] = -1
    b = np.array([-1]).reshape(1, )

    dsystem = DualLinearSystem(S, irrev, T, b)
    lsystem = IrreversibleLinearSystem(S, irrev)
    ksh = KShortestEnumerator(dsystem)

    done = False
    data = []
    while not done:
        try:
            sol,st = ksh.get_single_solution()
            isum = sol.attribute_value('indicator_sum')
            vsum = sol.attribute_value('var_sum')
            data.append(isum)
        except:
            done = True

    df = pd.DataFrame(data)
    df
    ksh.model.to_lp()
    ksh.model.problem.write('test_model.lp')
