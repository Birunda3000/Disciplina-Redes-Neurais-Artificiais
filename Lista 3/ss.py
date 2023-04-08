def degrau(entrada, limiar):
    if entrada > limiar:
        return 1
    else:
        return 0

def rede_neural(x, y, a1, a2, b1, b2, c1, c2, limiarA, limiarB, limiarC):

    saidaA = degrau(entrada=x * a1 + y * a2, limiar=limiarA)
    saidaB = degrau(entrada=x * b1 + y * b2, limiar=limiarB)
    saidaC = degrau(entrada=x * c1 + y * c2, limiar=limiarC)

    saidaRede = degrau(entrada=saidaA * 1 + saidaB * -1 + saidaC * 1, limiar=1)

    return saidaRede