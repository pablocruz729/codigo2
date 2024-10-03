
from fileTreat import *
import pyvista as pv #pip install pyvista
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc

# Carregando o arquivo VTK

#%%
run = '015'
path = fr"./../bin/TEST/{run}/"
steps = getMacrSteps(fr"./../bin/TEST/{run}/")# get macroscopics from last step
info = getSimInfo(path)
macro_save = 10000
Re=25000
L=info['NY']

N_steps = info['Nsteps']
Ulid = info['Umax']
NX = info['NX']
NY = info['NY']
NZ = info['NZ']
visc = (NX*Ulid)/Re
pasta = f'{run}_plots'
totalSteps = int(N_steps/macro_save)

# ti = 180 # step (macrosave) onde a energia cinetica para de crescer
ti = 60
tf = 399 # N_STEPS - 1

#%% CÁLCULO DA ENERGIA CINÉTICA


E = np.zeros(int(N_steps/macro_save))
E_adim = np.zeros(totalSteps)
cycles = np.arange(macro_save,N_steps+1,macro_save)
for i in range(cycles.size):
    cycle = cycles[i]
    filename = fr'..\bin\TEST\{run}\{run}macr{cycle:06d}.vtr'
    mesh = pv.read(filename)
    # Acessando os dados armazenados    
    uSum = np.array(mesh.point_data['uSum']) # ux²+uy²+uz²
    E[i] = np.sum(uSum)*0.5
    E_adim[i] = E[i]/((Ulid**2)*NY*NZ*NX)
    print(f'Cycle {i}/{cycles.size}')
    
#%% PLOT DA ENERGIA CINÉTICA
# totalKE = np.loadtxt(f"./../bin/TEST/{run}/{run}_totalKineticEnergy.txt") #pega a EC calculada dentro do código 
pasta = f'{run}_plots'
os.makedirs(pasta, exist_ok=True)
t_star = (cycles*Ulid)/L
np.savetxt(fr"Energia_Cinetica{NY}.csv", E_adim, delimiter=",")
np.savetxt(fr"t_star{NY}.csv", t_star, delimiter=",")
# Salvando o plot
plt.figure(figsize=(8,6),dpi=300)
plt.plot(t_star, E_adim,'k-')
plt.xlabel('$t*$')
plt.ylabel('$E*$')
plt.vlines(x=(cycles[ti]*Ulid)/L, ymin=np.min(E_adim), ymax=E_adim[ti], colors = 'b', linestyles='dashed')
plt.vlines(x=(cycles[tf]*Ulid)/L, ymin=np.min(E_adim), ymax=E_adim[tf], colors = 'b', linestyles='dashed')
plt.fill_between(t_star[ti:tf+1], E_adim[ti:tf+1], np.min(E_adim), where = E_adim[ti:tf+1] > 0, color='blue', interpolate = 'True', alpha = 0.3)
# plt.title('Energia Cinética por Passo de Tempo')
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_kinetic_energy.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()


#%%  CÁLCULO 
cyclesTurbulent = np.arange(macro_save*(ti+1),macro_save*(tf+1)+1,macro_save)
variablesMean = {}
uAlpha = np.zeros((tf-ti+1,L**3))
variable = ['ux','uy','uz','rho']
fluctuations = np.zeros((len(variable), cyclesTurbulent.size, L**3))
for j in range(len(variable)):
    variableManager = np.zeros((tf-ti+1,L**3))
    for i in range (cyclesTurbulent.size):
        cycle = cyclesTurbulent[i]
        filename = fr'..\bin\TEST\{run}\{run}macr{cycle:06d}.vtr'
        mesh = pv.read(filename)
        variableCatcher = np.array(mesh.point_data[variable[j]]) #armazena a variável de interesse
        variableManager[i,:] = variableCatcher[:] #armazena o array da variável para cada ciclo de interesse
        print(f'Cycle {i}/{cyclesTurbulent.size}')
    variablesMean[variable[j] + '_mean'] = np.mean(variableManager, axis=0)#calcula a média das variável de interesse (ux,uy,uz,rho,p)
    fluctuations[j, :, :] = variableManager - variablesMean[variable[j] + '_mean'] #array das flutuações (ux,uy,uz,rho,p)
    print(f'Loop variáveis {j}/{len(variable)}')

    del variableManager, variableCatcher #deleta variáveis intermediárias
    gc.collect()

uXMean = variablesMean['ux_mean']
uYMean = variablesMean['uy_mean']
uZMean = variablesMean['uz_mean']

del variablesMean



#%%  TENSOR TAXA DE DEFORMAÇÃO - Salphabeta
Salphabeta = np.zeros((6,L**3)) # tensor S ALPHA BET
arrayTransportTerm = np.zeros((9, cyclesTurbulent.size, L**3))
arrayProductionTerm = np.zeros((6, cyclesTurbulent.size, L**3))
arraySS = np.zeros((4, cyclesTurbulent.size, L**3))

for j in range (6): # LOOP EXTERNO PARA TRATAMENTO DE VARIÁVEIS
    componentCatcher = np.zeros((cyclesTurbulent.size, L**3)) # busca a componente de interesse (Sxx, Sxy...,etc)
    for i in range (cyclesTurbulent.size): # LOOP PARA INTERVALO DE INTERESSE (ESTATISTICAMENTE ESTACIONÁRIO)
        file_path = fr"./../bin/TEST/{run}/{run}_tensorSab{cyclesTurbulent[i]}.txt"
        data = []      
        with open(file_path, 'r') as file: # Abrindo e lendo o arquivo, ignorando linhas em branco
            for linha in file: 
                linha = linha.strip() # Remover espaços e quebras de linha, e ignorar linhas vazias
                if linha:
                    valores = [float(valor) for valor in linha.split()] # Converter a linha em uma lista de floats
                    data.append(valores)      
        data_array = np.array(data) # Converte a lista de listas em um numpy array
        n_variaveis = data_array.shape[1] # Verifica quantas variáveis (colunas) existem
        tensorSab = data_array.reshape(1, L**3, n_variaveis).transpose(2, 0, 1)
        componentCatcher[i,:] = tensorSab[j] 
        print(f'Cycle {i}/{cyclesTurbulent.size}')
    del data, data_array, valores, tensorSab
    print(f'Loop tensor {j}/6')
    Salphabeta[j,:] = np.mean(componentCatcher,axis=0) #obtenção do tensor Sab médio - <Sab> (<Sxx>, <Sxy>..., etc)
    if j == 0:
        arrayTransportTerm[j,:,:] = fluctuations[j,:,:]*componentCatcher  # u'x * Sxx
        arrayProductionTerm[j,:,:] = (fluctuations[j,:,:]**2)*componentCatcher  # (u'x)² * Sxx
        arraySS[j,:,:] = componentCatcher**2  # Sxx²
    if j == 1:
        arrayTransportTerm[j+2,:,:] = fluctuations[j-1,:,:]*componentCatcher  #  u'x * Sxy
        arrayTransportTerm[j,:,:] = fluctuations[j,:,:]*componentCatcher  #  u'y * Sxy
        arrayProductionTerm[j,:,:] = 2*(fluctuations[j,:,:]*fluctuations[j-1,:,:]*componentCatcher)  # 2*( u'x * u'y * Sxy)
        arraySS[j,:,:] += 2*(componentCatcher**2)  # Sxy²
    if j == 2:
        arrayTransportTerm[j,:,:] = fluctuations[j,:,:]*componentCatcher  # u'z * Sxz
        arrayTransportTerm[j+4,:,:] = fluctuations[j-2,:,:]*componentCatcher # u'x * Sxz
        arrayProductionTerm[j,:,:] = 2*(fluctuations[j-2,:,:]*fluctuations[j,:,:]*componentCatcher)  # 2*( u'x * u'z * Sxz)
        arraySS[j-1,:,:] += 2*(componentCatcher**2)  # Sxz²
    if j == 3:
        arrayTransportTerm[j+1,:,:] = fluctuations[j-2,:,:]*componentCatcher #  u'y*Syy
        arrayProductionTerm[j,:,:] = (fluctuations[j-2,:,:]**2)*componentCatcher #  u'y² * Syy
        arraySS[j-1,:,:] = componentCatcher**2  # Syy²
    if j == 4:
        arrayTransportTerm[j+1,:,:] = fluctuations[j-2,:,:]*componentCatcher #  u'z *Syz
        arrayTransportTerm[j+3,:,:] = fluctuations[j-3,:,:]*componentCatcher  # u'y * Syz
        arrayProductionTerm[j,:,:] = 2*(fluctuations[j-3,:,:]*fluctuations[j-2,:,:]*componentCatcher)
        arraySS[j-3,:,:] += 2*(componentCatcher**2)  # 2* Syz²
        
    if j == 5:
        arrayTransportTerm[j+3,:,:] = fluctuations[j-3,:,:]*componentCatcher # u'z * Szz
        arrayProductionTerm[j,:,:] = (fluctuations[j-3,:,:]**2)*componentCatcher #  u'z² * Szz
        arraySS[j-2,:,:] = componentCatcher**2  # Szz²
        
    del componentCatcher
production = np.sum(arrayProductionTerm,axis=0)
SS = np.sum(arraySS, axis=0)
del arrayProductionTerm, arraySS
meanProduction = np.mean(production,axis=0)
epsilon = 2*visc*np.mean(SS, axis=0)
del production






#%%

uFluctSquareSum = fluctuations[0,:,:]**2 + fluctuations[1,:,:]**2 + fluctuations[2,:,:]**2
rho_1 = np.ones(L**3)
# Cálculo do termo A de transporte - u'alpha u'beta u'beta
aTermX = fluctuations[0,:,:]*(uFluctSquareSum)
aTermY = fluctuations[1,:,:]*(uFluctSquareSum)
aTermZ = fluctuations[2,:,:]*(uFluctSquareSum)

aTermX = 0.5*(np.mean(aTermX, axis = 0)) # 1/2*<u'alpha u'beta u'beta>
aTermY = 0.5*(np.mean(aTermY, axis = 0))
aTermZ = 0.5*(np.mean(aTermZ, axis = 0))

#Cálculo do Termo B de transporte - <u'alpha * p'/rho>

bTermX = fluctuations[0,:,:] * (fluctuations[3,:,:]/rho_1)
bTermY = fluctuations[1,:,:] * (fluctuations[3,:,:]/rho_1)
bTermZ = fluctuations[2,:,:] * (fluctuations[3,:,:]/rho_1)

bTermX = np.mean(bTermX, axis = 0)
bTermY = np.mean(bTermY, axis = 0)
bTermZ = np.mean(bTermZ, axis = 0)

#Cálculo do termo C de transporte - -2visc<u'beta * Sab>

cTermX = arrayTransportTerm[0,:,:] + arrayTransportTerm[1,:,:] + arrayTransportTerm[2,:,:]
cTermY = arrayTransportTerm[3,:,:] + arrayTransportTerm[4,:,:] + arrayTransportTerm[5,:,:]
cTermZ = arrayTransportTerm[6,:,:] + arrayTransportTerm[7,:,:] + arrayTransportTerm[8,:,:]

cTermX = 2*visc*np.mean(cTermX, axis =0 )
cTermY = 2*visc*np.mean(cTermY, axis =0 )
cTermZ = 2*visc*np.mean(cTermZ, axis =0 )
        

# TERMO DE TRANSPORTE TURBULENTO 
tx = aTermX + bTermX - cTermX 
ty = aTermY + bTermY - cTermY
tz = aTermZ + bTermY - cTermZ

del aTermX, aTermY, aTermZ, bTermX, bTermY, bTermZ, arrayTransportTerm, cTermX, cTermY, cTermZ
gc.collect()


# ENERGIA CINÉTICA TURBULENTA (TKE)
uAlpha = np.mean(uFluctSquareSum, axis = 0)
turbulentKineticEnergy = uAlpha/2
turbulentKineticEnergy = turbulentKineticEnergy/(Ulid**2)
tKEVector = np.reshape(turbulentKineticEnergy, (NX,NY,NZ))


#%% APROXIMAÇÃO DERIVADAS
# deltaX = NX/L
# deltaY = NY/L
# deltaZ = NZ/L
dxkx = np.zeros(NX**3)
dyky = np.zeros(NY**3)
dzkz = np.zeros(NZ**3)
taux = np.zeros(NX**3)
tauy = np.zeros(NX**3)
tauz = np.zeros(NX**3)
count=0
l=0
for i in range(uYMean.size):
    
# VARREDURA DE DERIVADAS PARA d/d(x)
    if i < (NX**3)-1:
        dxkx[i] = (turbulentKineticEnergy[i+1] - turbulentKineticEnergy[i-1])/2
        taux[i] = (tx[i+1] - tx[i-1])/2
    if i%NX==0:
        dxkx[i] = (turbulentKineticEnergy[i+1] - turbulentKineticEnergy[i])
        taux[i] = (tx[i+1] - tx[i])
    if i%NX==(NX-1):
        dxkx[i] = (turbulentKineticEnergy[i] - turbulentKineticEnergy[i-1])
        taux[i] = (tx[i] - tx[i-1])
# VARREDURA DE DERIVADAS PARA d/d(y)
    if i%NY==0:
        count+=1   
    if count < NY: #VARREDURA PRINCIPAL
        if count ==1: #VARREDURA NA PARTE INFERIOR
            dyky[i]=(turbulentKineticEnergy[i+NY] - turbulentKineticEnergy[i])
            tauy[i]=(ty[i+NY] - ty[i])
        else:  #VARREDURA NA PARTE INTERNA
            dyky[i] = (turbulentKineticEnergy[i+NY] - turbulentKineticEnergy[i-NY])/2
            tauy[i] = (ty[i+NY] - ty[i-NY])/2
    else:
        if l < NX: # VARREDURA NA PARTE SUPERIOR
            dyky[i] = (turbulentKineticEnergy[i] - turbulentKineticEnergy[i-NY])/2
            tauy[i] = (ty[i] - ty[i-NY])/2
            l+=1
            if l == NX:
                count = 0
                l = 0 
# VARREDURA DE DERIVADAS PARA d/d(z)
    if 0 <= i < NZ**2:
        dzkz[i] = (turbulentKineticEnergy[i+NZ**2] - turbulentKineticEnergy[i])
        tauz[i] = (tz[i+NZ**2] - tz[i])
        
    if NZ**2 <= i < ((NZ**3)-NZ**2):
        dzkz[i] = (turbulentKineticEnergy[i+NZ**2] - turbulentKineticEnergy[i-NZ**2])/2
        tauz[i] = (tz[i+NZ**2] - tz[i-NZ**2])/2
        
    if ((NZ**3)-NZ**2) <= i < NX**3:
        dzkz[i] = (turbulentKineticEnergy[i-NZ**2] - turbulentKineticEnergy[i])
        tauz[i] = (tz[i-NZ**2] - tz[i])
    
kappaX = uXMean*dxkx
kappaY = uYMean*dyky
kappaZ = uZMean*dzkz










#%%   MÉDIA NOS PLANOS
kappaX = np.reshape(kappaX,(NX,NY,NZ))
kappaY = np.reshape(kappaY, (NX,NY,NZ))
kappaZ = np.reshape(kappaZ, (NX,NY,NZ))

taux = np.reshape(taux, (NX,NY,NZ))
tauy = np.reshape(tauy, (NX,NY,NZ))
tauz = np.reshape(tauz, (NX,NY,NZ))

production = np.reshape(meanProduction, (NX,NY,NZ))
epsilon = np.reshape(epsilon, (NX,NY,NZ))

# MÉDIAS BUDGET PLANO YZ
kappaXmean = np.mean(kappaX, axis=2)
kappaXmean = np.mean(kappaXmean, axis=1)
tauxmean = np.mean(taux, axis=2)
tauxmean = np.mean(tauxmean, axis=1)
productionx = np.mean(production,axis=2)
productionx = np.mean(productionx,axis=1)
epsilonx = np.mean(epsilon, axis=2)
epsilonx = np.mean(epsilonx, axis=1)

# MÉDIAS BUDGET PLANO XZ
kappaYmean = np.mean(kappaY, axis=2)
kappaYmean = np.mean(kappaYmean, axis=0)
tauymean = np.mean(tauy, axis=2)
tauymean = np.mean(tauymean, axis=0)
productiony = np.mean(production,axis=2)
productiony = np.mean(productiony,axis=0)
epsilony = np.mean(epsilon, axis=2)
epsilony = np.mean(epsilony, axis=0)

# # MÉDIAS BUDGET PLANO XY
kappaZmean = np.mean(kappaZ, axis=1)
kappaZmean = np.mean(kappaZmean, axis=0)
tauzmean = np.mean(tauz, axis=1)
tauzmean = np.mean(tauzmean, axis=0)
productionz = np.mean(production,axis=1)
productionz = np.mean(productionz,axis=0)
epsilonz = np.mean(epsilon, axis=1)
epsilonz = np.mean(epsilonz, axis=0)

del taux, tauy, tauz, production, epsilon, kappaX, kappaY, kappaZ


#%% PLOT NO PLANO YZ (ao longo de X)
x = np.linspace(0, 1, NX)
y = np.linspace(0, 1, NY)
z = np.linspace(0, 1, NZ)

plt.figure(figsize=(8,6),dpi=300)
plt.title('Budget of turbulence Re = 10000')
plt.plot(x, (tauxmean)*1e4, 'k-',label = r'$\tau$')
plt.plot(x,(kappaXmean), 'b-', label = r'$\kappa$')
plt.plot(x,(productionx)*1e3, 'g-', label = '$\tP$')
plt.plot(x,(epsilonx)*1e1, 'r-', label = r'$\epsilon$')
plt.text(-0.18, 0.75e-7, 'Gain', rotation=90, verticalalignment='center', fontsize=12)
plt.text(-0.18, -0.25e-7, 'Loss', rotation=90, verticalalignment='center', fontsize=12)
plt.xlabel('x', fontsize = 12)
# plt.ylabel('k(x)')
plt.legend()
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_budgetX.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()

#%% PLOT SOMATÓRIO PARCELAS (em X)

budgetx = tauxmean + (kappaXmean) - (productionx) + (epsilonx) 
plt.figure(figsize=(8,6),dpi=300)
plt.title('Turbulence terms summation for Re = 10,000')
plt.plot(x, budgetx, 'k-',label = r'$\overline{\beta}(x)$')
plt.xlabel('x', fontsize = 12)
plt.ylabel(r'$\overline{\beta}(x)$', fontsize = 12)
plt.legend()
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_somaX.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()

#%% PLOT NO PLANO XZ (ao longo de Y)
plt.figure(figsize=(8,6),dpi=300)
plt.title('Budget of turbulence Re = 10000')
plt.plot(x, tauymean*1e3, 'k-',label = r'$\tau$')
plt.plot(x,kappaYmean, 'b-', label = r'$\kappa$')
plt.plot(x,productiony*1e3, 'g-', label = '$\tP$')
plt.plot(x,epsilony, 'r-', label = r'$\epsilon$')
plt.text(-0.18, 2e-7, 'Gain', rotation=90, verticalalignment='center', fontsize=12)
plt.text(-0.18, -3e-7, 'Loss', rotation=90, verticalalignment='center', fontsize=12)
plt.xlabel('y', fontsize = 12)
plt.legend()
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_budgetY.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()

#%% PLOT SOMATÓRIO PARCELAS (em Y)

budgety = tauymean + kappaYmean - productiony + epsilony 
plt.figure(figsize=(8,6),dpi=300)
plt.title('Turbulence terms summation for Re = 10,000')
plt.plot(x, budgety, 'k-',label = r'$\overline{\beta}(y)$')
plt.xlabel('y', fontsize = 12)
plt.ylabel(r'$\overline{\beta}(y)$', fontsize = 12)
plt.legend()
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_somaY.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()




#%% PLOT NO PLANO XY (ao longo de Z)

plt.figure(figsize=(8,6),dpi=300)
plt.title('Budget of turbulence Re = 10000')
plt.plot(x, tauzmean*1e4, 'k-',label = r'$\tau$')
plt.plot(x,kappaZmean, 'b-', label = r'$\kappa$')
plt.plot(x,productionz*1e1, 'g-',label = '$\tP$')
plt.plot(x,epsilonz, 'r-' ,label = r'$\epsilon$')
plt.text(-0.18, 3e-8, 'Gain', rotation=90, verticalalignment='center', fontsize=12)
plt.text(-0.18, -2e-8, 'Loss', rotation=90, verticalalignment='center', fontsize=12)
plt.xlabel('z', fontsize = 12)
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_budgetZ.pdf')
plt.savefig(nome_arquivo_plot)
plt.legend()
plt.show()

#%% PLOT SOMATÓRIO PARCELAS (em Z)

budgetz = tauzmean + kappaZmean - productionz + epsilonz
plt.figure(figsize=(8,6),dpi=300)
plt.title('Turbulence terms summation for Re = 10,000')
plt.plot(x, budgetz, 'k-',label = r'$\overline{\beta}(z)$')
plt.xlabel('z', fontsize = 12)
plt.ylabel(r'$\overline{\beta}(z)$', fontsize = 12)
plt.legend()
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_somaZ.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()

        

#%% PLOT FLUTUAÇÃO DA VELOCIDADE 


uFluctVFluct = fluctuations[0,:,:]*fluctuations[1,:,:]
uFluctVFluct = np.mean(uFluctVFluct, axis= 0)

# # Média do quadrado da flutuação (128³ valores para cada dirção)
uXFluctMean = np.mean(fluctuations[0,:,:]**2, axis = 0) # <(u')²>
uYFluctMean = np.mean(fluctuations[1,:,:]**2, axis = 0) # <(v')²>
uZFluctMean = np.mean(fluctuations[2,:,:]**2, axis = 0) # <(w')²>
del fluctuations

uXFluctRoot = np.sqrt(uXFluctMean)
uYFluctRoot = np.sqrt(uYFluctMean)
uZFluctRoot = np.sqrt(uZFluctMean)

uRMSVector = np.reshape(uXFluctRoot, (NX,NY,NZ))
vRMSVector = np.reshape(uYFluctRoot, (NX,NY,NZ))
wRMSVector = np.reshape(uZFluctRoot, (NX,NY,NZ))
uFluctVFluct = np.reshape(uFluctVFluct, (NX,NY,NZ))


#%% PLOT URMS VRMS UPRIMEVPRIME 
uPrimeVPrimex = [(uFluctVFluct [int(NX/2), y, int(NZ/2)]) for y in range(0, NY)]
uPrimeVPrimey = [(uFluctVFluct [int(NX/2),int(NY/2), x]) for x in range(0, NX)]
uRMS = [(uRMSVector [int(NX/2), y, int(NZ/2)]) for y in range(0, NY)]
uRMS = np.divide(uRMS, Ulid)
uRMS = uRMS*10

vRMS = [(vRMSVector [int(NX/2), int(NY/2), z]) for z in range(0, NZ)]
vRMS = np.divide(vRMS, Ulid)
vRMS = vRMS*10

wRMS = [(wRMSVector [int(NX/2), x, int(NZ/2)]) for x in range(0, NX)]
wRMS = np.divide(wRMS, Ulid)
wRMS = wRMS*10


uPrimeVPrimex = np.divide(uPrimeVPrimex, Ulid**2)
uPrimeVPrimex = uPrimeVPrimex*500

uPrimeVPrimey = np.divide(uPrimeVPrimey, Ulid**2)
uPrimeVPrimey = uPrimeVPrimey*500



# REFERÊNIAS RE = 7.500 (PRASAD E KROEFF)
xrms7500 = [0.2573018080667593, 0.3518776077885952, 0.39082058414464527, 0.32684283727399155, 0.35744089012517377, 0.3546592489568845, 0.3852573018080667, 0.38247566063977745, 0.36022253129346304, 0.3296244784422808, 0.29346314325452005, 0.1905424200278163, 0.14325452016689844]
yrms7500 = [-0.9806629834254144, -0.9668508287292817, -0.9558011049723757, -0.9281767955801105, -0.9005524861878453, -0.861878453038674, -0.7955801104972375, -0.7292817679558011, -0.6629834254143647, -0.5966850828729282, -0.39226519337016574, -0.19613259668508287, 0.0027624309392264568]
xupvp7500 = [0.020862308762169546, 0.05146036161335177, -0.015299026425591111, -0.04867872044506261, -0.4075104311543811, -0.07093184979137701, -0.0931849791376913, -0.3018080667593881, -0.35187760778859534, -0.06258692628650908, -0.05146036161335188]
yupvp7500 = [-0.9640883977900553, -0.9337016574585635, -0.9088397790055249, -0.861878453038674, -0.8011049723756907, -0.7292817679558011, -0.6629834254143647, -0.5966850828729282, -0.40055248618784534, -0.19060773480662985, 0.0027624309392264568]
xupvpy7500 = [-0.9694019471488178, -0.9360222531293463, -0.9082058414464534, -0.8692628650904033, -0.8025034770514604, -0.7385257301808067, -0.6717663421418637, -0.6022253129346314, -0.40472878998609185, -0.2016689847009736, -0.0013908205841447474]
yupvpy7500 = [-0.002762430939226568, -0.06353591160220995, 0.013812154696132506, 0.022099447513812098, -0.005524861878453136, -0.03867403314917128, -0.013812154696132617, 0, -0.0303867403314918, -0.05524861878453047, 0.0055248618784529135]
xrmsy7500 = [-0.9582753824756607, -0.9749652294853964, -0.933240611961057, -0.9054242002781641, -0.866481223922114, -0.7969401947148818, -0.732962447844228, -0.6662030598052852, -0.5994436717663422, -0.3963838664812239, -0.19888734353268434, 0.0013908205841446364]
yrmsy7500 = [0.21270718232044183, 0.1325966850828728, 0.33149171270718236, 0.2624309392265194, 0.2596685082872927, 0.24861878453038666, 0.19613259668508287, 0.2541436464088398, 0.2541436464088398, 0.20165745856353579, 0.1629834254143645, 0.11602209944751385]
# REFERÊNCIAS RE=10.000 (PRASAD E KROEFF)
xrms10000 = [0.002873563218390718, 0.24425287356321834, 0.37356321839080464, 0.38505747126436773, 0.35632183908045967, 0.3793103448275863, 0.382183908045977, 0.43965517241379315, 0.48275862068965525, 0.38505747126436773, 0.3879310344827587, 0.28735632183908044, 0.16954022988505746, 0.14942528735632177, 0.12356321839080464, 0.09195402298850563, 0.12643678160919536, 0.1206896551724137, 0.13218390804597702, 0.14942528735632177, 0.15517241379310343, 0.14942528735632177, 0.1637931034482758, 0.3017241379310345, 0.36494252873563227, 0.4224137931034482, 0.008620689655172376]
yrms10000 = [-1, -0.9857142857142858, -0.9714285714285714, -0.96, -0.9314285714285714, -0.9057142857142857, -0.8628571428571429, -0.8, -0.7314285714285714, -0.6628571428571428, -0.5971428571428572, -0.39714285714285713, -0.19428571428571428, 0.005714285714285783, 0.20571428571428574, 0.4057142857142857, 0.6028571428571428, 0.6714285714285715, 0.7371428571428571, 0.8028571428571429, 0.8714285714285714, 0.9114285714285715, 0.937142857142857, 0.9628571428571429, 0.9742857142857142, 0.9828571428571429, 1]
xupvp10000  = [0, 0.02011494252873569, 0.03735632183908044, -0.04597701149425293, -0.1839080459770115, -0.2155172413793104, -0.4224137931034483, -0.3448275862068966, -0.26436781609195403, -0.24137931034482762, -0.2385057471264368, -0.04597701149425293, -0.014367816091954033, 0.005747126436781658, -0.005747126436781658, 0.014367816091954033, 0.014367816091954033, 0, -0.017241379310344862, 0.01724137931034475, 0.011494252873563315, -0.005747126436781658, -0.06896551724137934, -0.11206896551724144, 0.008620689655172376]
yupvp10000  = [-0.9885714285714285, -0.9742857142857143, -0.9571428571428572, -0.9314285714285714, -0.9057142857142857, -0.8685714285714285, -0.8, -0.7285714285714286, -0.6657142857142857, -0.6, -0.39714285714285713, -0.19714285714285718, 0.0028571428571428914, 0.20285714285714285, 0.4028571428571428, 0.6028571428571428, 0.6685714285714286, 0.7371428571428571, 0.8057142857142856, 0.8685714285714285, 0.9085714285714286, 0.937142857142857, 0.9657142857142857, 0.9771428571428571, 1]
xupvpy10000 = [-0.9827586206896551, -0.9540229885057472, -0.9281609195402298, -0.8994252873563219, -0.8591954022988506, -0.7931034482758621, -0.7270114942528736, -0.6637931034482758, -0.5948275862068966, -0.3936781609195402, -0.1954022988505747, 0.005747126436781658, 0.20977011494252862, 0.4051724137931034, 0.6091954022988506, 0.6752873563218391, 0.7413793103448276, 0.8103448275862069, 0.8706896551724137, 0.9137931034482758, 0.9396551724137931, 0.9683908045977012, 0.9770114942528736, 0.9971264367816091]
yupvpy10000 = [0.0028571428571428914, 0.008571428571428674, -0.03428571428571425, -0.03428571428571425, 0.037142857142857144, 0.014285714285714235, -0.03428571428571425, 0.03428571428571425, 0.011428571428571344, 0.02285714285714291, -0.05142857142857138, -0.03428571428571425, -0.017142857142857126, 0.011428571428571344, 0.040000000000000036, 0.07428571428571429, 0.11428571428571432, 0.08857142857142852, 0.4542857142857142, 0.5657142857142856, 0.3514285714285714, 0.011428571428571344, -0.005714285714285672, 0]
xrmsy10000 = [-0.9741379310344828, -0.9683908045977011, -0.9511494252873564, -0.9224137931034483, -0.896551724137931, -0.8591954022988506, -0.7902298850574713, -0.7270114942528736, -0.6580459770114943, -0.5948275862068966, -0.3936781609195402, -0.19252873563218387, 0.005747126436781658, 0.20977011494252862, 0.40804597701149414, 0.6063218390804597, 0.6724137931034482, 0.7385057471264367, 0.8074712643678161, 0.8764367816091954, 0.9425287356321839, 0.9626436781609196, 0.9827586206896552, 0.9942528735632183, 1.0057471264367814]
yrmsy10000 = [0.10000000000000009, 0.14571428571428569, 0.22857142857142865, 0.27714285714285714, 0.24285714285714288, 0.2142857142857142, 0.1942857142857144, 0.1942857142857144, 0.2142857142857142, 0.18571428571428572, 0.17714285714285705, 0.22571428571428576, 0.20285714285714285, 0.14857142857142858, 0.09142857142857141, 0.10000000000000009, 0.12857142857142856, 0.20285714285714285, 0.23142857142857132, 0.72, 0.9857142857142858, 0.3799999999999999, 0.36571428571428566, 0.1399999999999999, 0.0028571428571428914]



xt = [-0.9582753824756607, -0.9749652294853964, -0.933240611961057, -0.9054242002781641, -0.866481223922114, -0.7969401947148818, -0.732962447844228, -0.6662030598052852, -0.5994436717663422, -0.3963838664812239, -0.19888734353268434, 0.004172461752433909]
yt = [0.606353591160221, 0.5662983425414365, 0.6657458563535912, 0.6312154696132597, 0.6298342541436464, 0.6243093922651933, 0.5980662983425414, 0.6270718232044199, 0.6270718232044199, 0.600828729281768, 0.5814917127071824, 0.5552486187845304]

fig, ax1 = plt.subplots(figsize = (8,6), dpi=300)

ax2=ax1.twinx()
ax3=ax1.twiny() 
x = np.linspace(-1, 1, NX)
y = np.linspace(-1, 1, NY)
urmsx, = ax1.plot(xrms10000, yrms10000,'rx', alpha = 1)
upvpx, = ax1.plot(xupvp10000, yupvp10000,'b+', alpha = 1)
upvpyref, = ax2.plot(xupvpy10000, yupvpy10000,'g+', alpha = 1)
vrmsy, = ax2.plot(xrmsy10000, yrmsy10000,'cx', alpha = 1)
# urmsx, = ax1.plot(xrms7500, yrms7500,'ro-',alpha = 0.3)
# upvpx, = ax1.plot(xupvp7500, yupvp7500,'bo-',alpha = 0.3)
# upvpyref, = ax3.plot(xupvpy7500, yupvpy7500,'go-',alpha = 0.3)
# vrmsy, = ax3.plot(xrmsy7500, yrmsy7500,'co-',alpha = 0.3)
urms, = ax1.plot(uRMS, y, 'r-')
# urms, = ax1.plot(wRMS, y, 'r-')
upvp, = ax1.plot(uPrimeVPrimex, y, 'b-', label = "Velocidade")
upvpy, = ax1.plot(x,uPrimeVPrimey, 'g--')
vrms,= ax1.plot(x, vRMS, 'c--')
xtickloc = ax1.get_xticks() 
ytickloc = ax1.get_yticks()
ax1.set_xlabel(r'$u^{rms}_{x}$, $\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$')
ax1.set_ylabel(r'$u^{rms}_{y}$, $\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$')
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
# ax1.set_ylabel('y')
# ax2.set_ylabel('$u_y$')
# ax3.set_xlabel('x')
plt.hlines(y=0, xmin=-1, xmax=1, color= 'grey')
plt.vlines(x=0, ymin=-1, ymax=1, color='grey')
ax3.set_xticks(xtickloc)
ax3.set_xlim([-1,1])
ax2.set_ylim([-1,1])
plt.legend([urms,urmsx,upvp,upvpx,upvpy,upvpyref, vrms, vrmsy],
["$u^{rms}_{x}$ LBM","$u^{rms}_{x}$ Prasad", r"$\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$ LBM", r"$\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$ Prasad", 
r"$\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$ LBM",r"$\langle u^{\prime}_{x}u^{\prime}_{y}\rangle$ Prasad", "$u^{rms}_{y}$ LBM", "$u^{rms}_{y}$ Prasad"], loc = "lower left")
# plt.title('Perfis de' r'$\quad$' r'$10 \cdot RMS$' r'$\quad$' 'e' r'$\quad$' r'$500 \cdot \langle u^{\prime}_{x}u^{\prime}_{y}\rangle$' r'$\quad$' 'para Re = 10000')
nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_perfis_rms.pdf')
plt.savefig(nome_arquivo_plot)
plt.show()


#%% PLOT FLUTUAÇÃO DA VELOCIDADE PARA CADA DIREÇÃO, ENFATIZANDO A QUESTÃO DA MÉDIA ZERAR ANTES DE ELEVAR AO QUADRADO
# t_star_turbulent = (cyclesTurbulent*Ulid)/L

# # Média da flutuação em cada time step (tf-ti+1 valores para cada direção)
# uXFluctTime = np.mean(uXFluct, axis = 1)
# uYFluctTime = np.mean(uYFluct, axis = 1)
# uZFluctTime = np.mean(uZFluct, axis = 1)


# plt.figure(dpi=300)
# plt.plot(t_star_turbulent,(uXFluctTime)/Ulid,'b')
# plt.hlines(y=0, xmin=np.min(t_star_turbulent), xmax=np.max(t_star_turbulent), colors='k')
# plt.xlabel('$t^{*}$')
# plt.xlim(xmin=np.min(t_star_turbulent),xmax=np.max(t_star_turbulent))
# plt.fill_between(t_star_turbulent, (uXFluctTime/Ulid), 0, where = uXFluctTime > 0, color='b', interpolate = 'True')
# plt.fill_between(t_star_turbulent, (uXFluctTime/Ulid), 0, where = uXFluctTime < 0, color='b', interpolate = 'True', alpha=0.5)
# plt.ylabel(r'$u_{x}^{\prime}/U_{lid}$')
# plt.ylim(-4e-5, 3e-5)
# # plt.title('Velocity fluctuations $u^{\prime}_{x}$')
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_fluctu_u.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()


# plt.figure(dpi=300)
# plt.plot(t_star_turbulent,(uYFluctTime)/Ulid,'purple')
# plt.hlines(y=0, xmin=np.min(t_star_turbulent), xmax=np.max(t_star_turbulent), colors='k')
# plt.xlabel('$t^{*}$')
# plt.xlim(xmin=np.min(t_star_turbulent),xmax=np.max(t_star_turbulent))
# plt.fill_between(t_star_turbulent, (uYFluctTime/Ulid), 0, where = uYFluctTime > 0, color='purple', interpolate = 'True')
# plt.fill_between(t_star_turbulent, (uYFluctTime/Ulid), 0, where = uYFluctTime < 0, color='purple', interpolate = 'True', alpha = 0.5)
# plt.ylabel(r'$u_{y}^{\prime}/U_{lid}$')

# plt.ylim(-4e-5, 3e-5)
# # plt.title('Velocity fluctuations $u^{\prime}_{y}$')
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_fluctu_v.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()


# plt.figure(dpi=300)
# plt.plot(t_star_turbulent,(uZFluctTime)/Ulid,'g')
# plt.hlines(y=0, xmin=np.min(t_star_turbulent), xmax=np.max(t_star_turbulent), colors='k')
# plt.xlabel('$t^{*}$')
# plt.xlim(xmin=np.min(t_star_turbulent),xmax=np.max(t_star_turbulent))
# plt.fill_between(t_star_turbulent, (uZFluctTime/Ulid), 0, where = uZFluctTime > 0, color='g', interpolate = 'True')
# plt.fill_between(t_star_turbulent, (uZFluctTime/Ulid), 0, where = uZFluctTime < 0, color='g', interpolate = 'True', alpha = 0.5)
# plt.ylabel(r'$u_{z}^{\prime}/U_{lid}$')
# plt.ylim(-4e-5, 3e-5)
# # plt.title('Velocity fluctuations $u^{\prime}_{z}$')
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_fluctu_w.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()



 

#%% PLOT ENERGIA CINÉTICA TURBULENTA - AO LONGO DE UM SLICE (X, Y OU Z)


# turbulentKinecticEnergy = turbulentKinecticEnergy[::20]
# y_teste = np.linspace (0,1,turbulentKinecticEnergy.size)
# plt.figure(dpi=300)
# plt.plot(y_teste,turbulentKinecticEnergy,'k')
# plt.xlabel('x')
# plt.ylabel('$E_c$')
# plt.show()
# tKESlice_01 = [(tKEVector [int(NX/10), int(NY/2), z]) for z in range(0, NZ)]
# tKESlice_05 = [(tKEVector [int(NX/2), int(NY/2), z]) for z in range(0, NZ)] 
# tKESlice_09 = [(tKEVector [int(NX/1.11), int(NY/2), z]) for z in range(0, NZ)]


# PLOTAR TKE EM DIFERENTES POSIÇÕES DE "Z", AO LONGO DE "X" EM Y = 0.5
# tKESlice_01 = [(tKEVector [x, int(NY/2), int(NZ/10)]) for x in range(0, NX)]
# tKESlice_05 = [(tKEVector [x, int(NY/2), int(NZ/2)]) for x in range(0, NX)] 
# tKESlice_09 = [(tKEVector [x, int(NY/2), int(NZ/1.11)]) for x in range(0, NX)]
# x = np.linspace(0, 1, NY)
# plt.figure(dpi=300)
# plt.plot(x, tKESlice_01,'k', label = '$X=0.1$')
# plt.plot(x, tKESlice_05,'b', label = '$X=0.5$')
# plt.plot(x, tKESlice_09,'r', label = '$X=0.9$')
# plt.legend()
# plt.xlabel('Z')
# plt.ylabel('$TKE$')
# # plt.title('Perfis de $TKE$ ao longo de $Z$')
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_profiles_along_z.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()







#%% PLOT CONTOUR EM UM CORTE ESPECÍFICO

# x = np.linspace(0, 1, NX)
# y = np.linspace(0, 1, NY)
# X,Y = np.meshgrid(x,y)



# # PLOT CONTOUR ENERGIA CINÉTICA TURBULENTA (TKE) NO PLANO X = 0.5
# levels = np.linspace(0.000,0.010,11)
# tKESliceContour = [(tKEVector [int(NX/2), :, :])]
# tKESliceTeste = np.reshape(tKESliceContour, (NX,NY))
# plt.figure(dpi=300)
# CS = plt.contour(x,y, tKESliceTeste,100,cmap = 'jet',levels = levels)
# plt.xlabel('Z')
# plt.ylabel('Y')
# cbar = plt.colorbar(CS)
# # plt.title('$TKE$ em $X = 0.5$')
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_TKE_contour.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()



# # PLOT CONTOUR u'_{rms} 
# levels = np.linspace(0.0000,0.0705,15)
# uRMSVectorContour = [(uRMSVector [int(NX/2), :, :])]
# uRMSVectorSlice = np.reshape(uRMSVectorContour, (NX,NY))
# uRMSVectorSlice = np.divide(uRMSVectorSlice, Ulid)
# plt.figure(dpi=300)
# CS = plt.contour(x,y, uRMSVectorSlice,100,cmap = 'jet',levels = levels)
# plt.xlabel('Z')
# plt.ylabel('Y')
# cbar = plt.colorbar(CS)
# title = r"$\sqrt{\langle\left(u^{\prime}\right)^{2}\rangle}$ em $x=0.5$"
# plt.title(title) 
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_root_uprime_square_contour.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()


# # PLOT CONTOUR v'_{rms} 
# levels = np.linspace(0.0000,0.1180,15)
# vRMSVectorContour = [(vRMSVector [int(NX/2), :, :])]
# vRMSVectorSlice = np.reshape(vRMSVectorContour, (NX,NY))
# vRMSVectorSlice = np.divide(vRMSVectorSlice, Ulid)
# plt.figure(dpi=300)
# CS = plt.contour(x,y, vRMSVectorSlice,75,cmap = 'jet',levels = levels)
# plt.xlabel('Z')
# plt.ylabel('Y')
# cbar = plt.colorbar(CS)
# title = r"$\sqrt{\langle\left(v^{\prime}\right)^{2}\rangle}$ em $x=0.5$"
# plt.title(title) 
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_root_vprime_square_contour.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()


# # PLOT CONTOUR u'v' 
# levels = np.linspace(-0.0016,0.0056,18)
# UpVpVectorContour = [(uFluctVFluct [int(NX/2), :, :])]
# UpVpVectorSlice = np.reshape(UpVpVectorContour, (NX,NY))
# UpVpVectorSlice = np.divide(UpVpVectorSlice, Ulid)
# plt.figure(dpi=300)
# CS = plt.contour(x,y, UpVpVectorSlice,100,cmap = 'jet')
# plt.xlabel('Z')
# plt.ylabel('Y')
# cbar = plt.colorbar(CS)
# title = r"$\langle\left(u^{\prime}v^{\prime}\right)\rangle$ em $x=0.5$"
# plt.title(title) 
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_uprime_vprime_contour.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()


# # PLOT CONTOUR u'v' 
# levels = np.linspace(-0.0016,0.0056,18)
# UpVpVectorContour = [(uFluctVFluct [int(NX/2), :, :])]
# UpVpVectorSlice = np.reshape(UpVpVectorContour, (NX,NY))
# UpVpVectorSlice = np.divide(UpVpVectorSlice, Ulid)
# plt.figure(dpi=300)
# CS = plt.contour(x,y, UpVpVectorSlice,100,cmap = 'jet')
# plt.xlabel('Z')
# plt.ylabel('Y')
# cbar = plt.colorbar(CS)
# title = r"$\langle\left(u^{\prime}v^{\prime}\right)\rangle$ em $x=0.5$"
# plt.title(title) 
# nome_arquivo_plot = os.path.join(pasta, f'{Re}_{run}_uprime_vprime_contour.pdf')
# plt.savefig(nome_arquivo_plot)
# plt.show()





    
