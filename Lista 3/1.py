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

'''
Defina valores para a1, a2, b1, b2, c1, c2, limiarA, limiarB, limiarC tal que a saida da rede seja 1 se x>0, y>0 e x+y<1 e 0 caso contrario.
Por exemplo:
caso x-0.5 e y=0.4, a saida da rede deve ser 1,
se x=1 e y=1 a saida da rede deve ser 0, 
se x=-1 e y=0,5 a saida da rede deve ser 0,
se x=0.1 e y=0.1 a saida da rede deve ser 1.
se x=0.1 e y=0.9 a saida da rede deve ser 1.
se x=0.9 e y=0.1 a saida da rede deve ser 1.
se x=0.9 e y=0.9 a saida da rede deve ser 0.
se x=0.5 e y=-0.5 a saida da rede deve ser 0.
se x=-0.5 e y=0.5 a saida da rede deve ser 0.
se x=-0.5 e y=-0.5 a saida da rede deve ser 0.
se x=5 e y=-2 a saida da rede deve ser 0.
'''
