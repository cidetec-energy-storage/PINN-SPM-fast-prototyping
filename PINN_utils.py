import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import scipy
import warnings
warnings.filterwarnings('ignore')
#

U_model_n = tf.keras.models.load_model('SPM NE')
U_model_p = tf.keras.models.load_model('SPM PE')
#
# reading the data from the file. I select NMC811 and OCVs used in that json.
with open('data/params_GrSi_NMC811.json') as f:
    data = f.read()
# reconstructing the data as a dictionary
parameters = json.loads(data)
#
R = tf.constant(parameters['constants']['R']['value'], dtype=tf.float64)
F = tf.constant(parameters['constants']['F']['value'], dtype=tf.float64)
alpha = tf.constant(parameters['constants']['alpha']['value'], dtype=tf.float64)
# negative electrode
L_n = parameters['negativeElectrode']['thickness']['value']
A_n = parameters['negativeElectrode']['area']['value']
epsilon_n = tf.constant(parameters['negativeElectrode']['porosity']['value'], dtype=tf.float64)
sigma_n = tf.constant(parameters['negativeElectrode']['electronicConductivity']['value'], dtype=tf.float64)
k_0_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['kineticConstant']['value'], dtype=tf.float64)
c_s_max_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['maximumConcentration']['value'], dtype=tf.float64)
R_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['particleRadius']['value'], dtype=tf.float64)
epsilon_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['volFrac_active']['value'], dtype=tf.float64)
D_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['diffusionConstant']['value'], dtype=tf.float64)
# separator
L_s = parameters['separator']['thickness']['value']
epsilon_sep = tf.constant(parameters['separator']['porosity']['value'], dtype=tf.float64)
# positive electrode
L_p = parameters['positiveElectrode']['thickness']['value']
A_p = parameters['positiveElectrode']['area']['value']
epsilon_p = tf.constant(parameters['positiveElectrode']['porosity']['value'], dtype=tf.float64)
sigma_p = tf.constant(parameters['positiveElectrode']['electronicConductivity']['value'], dtype=tf.float64)
k_0_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['kineticConstant']['value'], dtype=tf.float64)
c_s_max_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['maximumConcentration']['value'], dtype=tf.float64)
R_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['particleRadius']['value'], dtype=tf.float64)
epsilon_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['volFrac_active']['value'], dtype=tf.float64)
D_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['diffusionConstant']['value'], dtype=tf.float64)

c_e_0 = tf.constant(parameters['electrolyte']['initialConcentration']['value'], dtype=tf.float64)
#
# I select stoichiometry 1 in order to calculate initial concentrations and voltages.
stoi_1_n = parameters['negativeElectrode']['active_materials'][0]['stoichiometry1']['value']
stoi_0_n = parameters['negativeElectrode']['active_materials'][0]['stoichiometry0']['value']

stoi_1_p = parameters['positiveElectrode']['active_materials'][0]['stoichiometry1']['value']
stoi_0_p = parameters['positiveElectrode']['active_materials'][0]['stoichiometry0']['value']

# to calculate c_s_p_0 and c_s_n_0 as function of z (SoC):
z = 1
c_s_n_0 = tf.constant(c_s_max_n * (z*(stoi_1_n-stoi_0_n) + stoi_0_n), dtype=tf.float64)
c_s_p_0 = tf.constant(c_s_max_p * (z*(stoi_1_p-stoi_0_p) + stoi_0_p), dtype=tf.float64)
x_n_0 = c_s_n_0/c_s_max_n
x_p_0 = c_s_p_0/c_s_max_p
#
path_OCP_negative_elec = parameters['negativeElectrode']['active_materials'][0]['openCircuitPotential']['value']
OCP_anode = np.loadtxt('data/' + path_OCP_negative_elec)
inter_anode = scipy.interpolate.interp1d(OCP_anode[:, 0], OCP_anode[:, 1], fill_value='extrapolate')

path_OCP_positive_elec = parameters['positiveElectrode']['active_materials'][0]['openCircuitPotential']['value']
OCP_cathode = np.loadtxt('data/' + path_OCP_positive_elec)
inter_cathode = scipy.interpolate.interp1d(OCP_cathode[:, 0], OCP_cathode[:, 1], fill_value='extrapolate')
#
fracInMat_n = 1 - epsilon_n - epsilon_s_n #innactive material, suppose it is constant
fracInMat_p = 1 - epsilon_p - epsilon_s_p
#
Q_neg = A_n*F*L_n*epsilon_s_n*c_s_max_n*abs(stoi_1_n-stoi_0_n) # As #/3600 Ah
Q_pos = A_p*F*L_p*epsilon_s_p*c_s_max_p*abs(stoi_1_p-stoi_0_p) #/3600
Q = min(Q_neg, Q_pos)
C_rate = 1
switch_ch_disch = 1
I = Q/3600*C_rate*switch_ch_disch
L_0 = L_n + L_s + L_p
T = 298.15
#
cte_n = D_s_n/R_s_n**2
cte_p = D_s_p/R_s_p**2
bc_n = -R_s_n**2*I/(D_s_n*3*epsilon_s_n*L_n*c_s_max_n*F*A_n)
bc_p = R_s_p**2*I/(D_s_p*3*epsilon_s_p*L_p*c_s_max_p*F*A_p)
bc_p_limit = 1.769023333333332934e-01
#
def obtain_bcs(thickness_n, thickness_p, porosity_n, porosity_p, C_rate, 
               obtain_real_parameters=0, ne_or_pe_for_real_parameter='ne', min_value_bc=-0.012, max_value_bc=-0.010):
    epsilon_s_n_new = 1 - fracInMat_n - porosity_n
    epsilon_s_p_new = 1 - fracInMat_p - porosity_p

    Q_neg = A_n*F*thickness_n*epsilon_s_n_new*c_s_max_n*abs(stoi_1_n-stoi_0_n) 
    Q_pos = A_p*F*thickness_p*epsilon_s_p_new*c_s_max_p*abs(stoi_1_p-stoi_0_p)

    Q = min(Q_neg, Q_pos)
    I = Q/3600*C_rate

    bc_n = -R_s_n**2*I/(D_s_n*3*epsilon_s_n_new*thickness_n*c_s_max_n*F*A_n)
    bc_p = R_s_p**2*I/(D_s_p*3*epsilon_s_p_new*thickness_p*c_s_max_p*F*A_p)

    if obtain_real_parameters == True:
        if ne_or_pe_for_real_parameter == 'ne':
            if bc_n > min_value_bc and bc_n < max_value_bc:
                return thickness_n, thickness_p, porosity_n, porosity_p, C_rate
        if ne_or_pe_for_real_parameter == 'pe':
            if bc_p > min_value_bc and bc_p < max_value_bc:
                return thickness_n, thickness_p, porosity_n, porosity_p, C_rate

    else:
        return bc_n, bc_p, I

#
def normalize_Inputs_for_OCV(Inputs, Inputs_lim):
        t_norm = Inputs[:, 0]
        bc_norm = (Inputs[:, 2]-min(Inputs_lim[:, 2]))/(max(Inputs_lim[:, 2])-min(Inputs_lim[:, 2]))
        t_max_arr = (Inputs[:, 3]-min(Inputs_lim[:, 3]))/(max(Inputs_lim[:, 3])-min(Inputs_lim[:, 3]))
        Inputs_norm = np.stack((t_norm, Inputs[:, 1], bc_norm, t_max_arr), axis=1)
        return Inputs_norm
    
#
def create_input_from_bc(ne_or_pe, bc):
    if ne_or_pe == 'ne':
        h = abs(bc/bc_n)
        time_steps = 100
        r_steps = 32
        t_max = 3900/h
        # if t_max < 100*3600:
        #     t_max = 3600/h
        # else:
        #     t_max = 100*3600
        time = np.linspace(0, t_max, time_steps)/t_max
        r_arr_n = np.linspace(0.2e-6, R_s_n, r_steps)/R_s_n.numpy()
        Input = []
        for t_i, t in enumerate(time):
            for r_i, r in enumerate(r_arr_n):
                elem = [t, r, bc, t_max]
                Input.append(elem)
    if ne_or_pe == 'pe':
        h = abs(bc/bc_p_limit)
        time_steps = 100
        r_steps = 32
        t_max = 3900/h
        # if t_max < 100*3600:
        #     t_max = 3600/h
        # else:
        #     t_max = 100*3600
        time = np.linspace(0, t_max, time_steps)/t_max
        r_arr_p = np.linspace(0.2e-6, R_s_p, r_steps)/R_s_p.numpy()
        Input = []
        for t_i, t in enumerate(time):
            for r_i, r in enumerate(r_arr_p):
                elem = [t, r, bc, t_max]
                Input.append(elem)
    return np.array(Input)

#
bc_n_space = np.linspace(-0.011509128654707082, 3*bc_n, 2)
#
bc_p_space = np.linspace(bc_p*1e-1, 3.33*bc_p, 2)
#
Inputs_n = create_input_from_bc('ne', bc_n)
for bc_n_elem in bc_n_space: 
    input_elem_n = create_input_from_bc('ne', bc_n_elem)
    Inputs_n = np.concatenate((Inputs_n, input_elem_n), 0)
Inputs_n = np.delete(Inputs_n, np.s_[0:len(create_input_from_bc('ne', bc_n))], 0)

Inputs_p = create_input_from_bc('pe', bc_p)
for bc_p_elem in bc_p_space:
    input_elem_p = create_input_from_bc('pe', bc_p_elem)
    Inputs_p = np.concatenate((Inputs_p, input_elem_p), 0)
Inputs_p = np.delete(Inputs_p, np.s_[0:len(create_input_from_bc('pe', bc_p))], 0)
#
def obtain_position(value_norm, variable, Input):
    if variable == 'time':
        positions = []
        for pos, elem in enumerate(Input[:, 0]):
            if elem == value_norm:
                pos_to_save = pos
                positions.append(pos_to_save)
    if variable == 'space':
        positions = []
        for pos, elem in enumerate(Input[:, 1]):
            if elem == value_norm:
                pos_to_save = pos
                positions.append(pos_to_save)
    return positions
#
def obtain_x_n(select_or_input, n, input, t_max_n_input=0):
    if select_or_input == 'select':
        pass
    if select_or_input == 'input':
        Input = input
        t_max = t_max_n_input
    position_frontier_r_max = obtain_position(Input[:, 1][-1], 'space', Input)
    x_barra_n = tf.squeeze(tf.cast(U_model_n(Input), tf.float64))
    x_n = x_barra_n + x_n_0
    x_n_surface = []
    for position in position_frontier_r_max:
        x_n_surface.append(x_n[position])
    x_n_surface = np.array(x_n_surface)
    # eta_n = 2*R*T/F*np.arcsinh(R_s_n*I/(2*k_0_n*np.sqrt(c_e_0)*3*epsilon_s_n*L_n*c_s_max_n*F*A_n*np.sqrt(x_n_surface*(1-x_n_surface))))
    
    t_len = 100
    r_steps = 32
    time_arr = np.linspace(0, 1, t_len)
    r_arr_n = np.linspace(0.2e-6, R_s_n, r_steps)/R_s_n.numpy()
    Total_predicted_n = tf.squeeze(np.array(np.array_split(x_n, t_len)).T)
    plt.figure()
    plt.pcolormesh(time_arr*t_max, r_arr_n*R_s_n, Total_predicted_n*c_s_max_n)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$r_{NE}$ [m]')
    clb = plt.colorbar()
    clb.ax.set_ylabel(r'$c_{s,NE}$ [mol/m**3]')
    plt.title('NE')
    plt.xticks(fontsize=9, rotation=0)
    plt.yticks(fontsize=9, rotation=0)
    
    return x_n, x_n_surface, time_arr, t_max

#
def obtain_x_p(select_or_input, n, input, t_max_p_input=0):
    if select_or_input == 'select':
        pass
    if select_or_input == 'input':
        Input = input
        t_max = t_max_p_input
    position_frontier_r_max = obtain_position(Input[:, 1][-1], 'space', Input)
    x_barra_p = tf.squeeze(tf.cast(U_model_p(Input), tf.float64))
    x_p = x_barra_p + x_p_0
    x_p_surface = []
    for position in position_frontier_r_max:
        x_p_surface.append(x_p[position])
    x_p_surface = np.array(x_p_surface)
    # eta_p = 2*R*T/F*np.arcsinh(-R_s_p*I/(2*k_0_p*np.sqrt(c_e_0)*3*epsilon_s_p*L_p*c_s_max_p*F*A_p*np.sqrt(x_p_surface*(1-x_p_surface))))
    t_len = 100
    r_steps = 32
    time_arr = np.linspace(0, 1, t_len)
    r_arr_p = np.linspace(0.2e-6, R_s_p, r_steps)/R_s_p.numpy()
    Total_predicted_p = tf.squeeze(np.array(np.array_split(x_p, t_len)).T)
    plt.figure()
    plt.pcolormesh(time_arr*t_max, r_arr_p*R_s_p, Total_predicted_p*c_s_max_p)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$r_{PE}$ [m]')
    clb = plt.colorbar()
    clb.ax.set_ylabel(r'$c_{s,PE}$ [mol/m**3]')
    plt.title('PE')
    plt.xticks(fontsize=9, rotation=0)
    plt.yticks(fontsize=9, rotation=0)
    # plt.tick_params('both', length=6, width=2)
    
    return x_p, x_p_surface, time_arr, t_max

#
def plot_Battery(thickness_n=L_n, thickness_p=L_p, porosity_n=epsilon_n, porosity_p=epsilon_p, C_rate=1, experimental_t_or_f=False, exp=0):
    bc_n_OCV, bc_p_OCV, I = obtain_bcs(thickness_n, thickness_p, porosity_n, porosity_p, C_rate)
    
    if abs(bc_n_OCV) >= abs(Inputs_n[:, 2][0]) and abs(bc_p_OCV) >= abs(Inputs_p[:, 2][0]):
        
        epsilon_s_n_for_OCV = 1-fracInMat_n-porosity_n
        epsilon_s_p_for_OCV = 1-fracInMat_p-porosity_p
        
        Input_n_OCV = create_input_from_bc('ne', bc_n_OCV)
        Input_p_OCV = create_input_from_bc('pe', bc_p_OCV)
        
        t_max_n = Input_n_OCV[:, 3][0]
        t_max_p = Input_p_OCV[:, 3][0]
        
        time_arr_n = np.linspace(0, 1, 100)*t_max_n
        time_arr_p = np.linspace(0, 1, 100)*t_max_p
        
        Input_n_OCV_norm = normalize_Inputs_for_OCV(Input_n_OCV, Inputs_n)
        Input_p_OCV_norm = normalize_Inputs_for_OCV(Input_p_OCV, Inputs_p)
        
        # values
        input_n_for_ocv = obtain_x_n('input', 0, Input_n_OCV_norm, t_max_n)
        input_p_for_ocv = obtain_x_p('input', 0, Input_p_OCV_norm, t_max_p)
        
        x_n_surface = input_n_for_ocv[1]
        x_p_surface = input_p_for_ocv[1]
        
        x_n_surf_inter = scipy.interpolate.interp1d(time_arr_n, x_n_surface, fill_value='extrapolate')
        x_p_surf_inter = scipy.interpolate.interp1d(time_arr_p, x_p_surface, fill_value='extrapolate')
        
        t_total = min([t_max_n, t_max_p])
        t_arr_for_OCV = np.linspace(0, t_total, 100)
        
        x_n_surf_for_OCV = x_n_surf_inter(t_arr_for_OCV)
        x_p_surf_for_OCV = x_p_surf_inter(t_arr_for_OCV)
        
        eta_n_for_OCV = 2*R*T/F*np.arcsinh(R_s_n*I/(2*k_0_n*np.sqrt(c_e_0)*3*epsilon_s_n_for_OCV*L_n*c_s_max_n*F*A_n*np.sqrt(x_n_surf_for_OCV*(1-x_n_surf_for_OCV))))
        eta_p_for_OCV = 2*R*T/F*np.arcsinh(-R_s_p*I/(2*k_0_p*np.sqrt(c_e_0)*3*epsilon_s_p_for_OCV*L_p*c_s_max_p*F*A_p*np.sqrt(x_p_surf_for_OCV*(1-x_p_surf_for_OCV))))

        V = inter_cathode(x_p_surf_for_OCV) - inter_anode(x_n_surf_for_OCV) + eta_p_for_OCV - eta_n_for_OCV
        plt.figure()
        plt.plot(t_arr_for_OCV, V, '-', linewidth=1.5, color='blue') #, label='PINN SPM')
        if experimental_t_or_f == True:
            plt.plot(exp['# Time [s]'], exp['Voltage [V]'], 'r', label='FEM SPM', alpha=1)
        plt.xlabel(r'$t$ [s]')
        plt.ylabel(r'$Voltage$ [V]')
        thickness_n_um = thickness_n*1e6
        thickness_p_um = thickness_p*1e6
        plt.plot(0,3.4, 'w', label=r'$thickness_{NE} = %0.2f$' %thickness_n_um + ' µm')
        plt.plot(0,3.4, 'w', label=r'$thickness_{PE} = %0.2f$' %thickness_p_um + ' µm')
        plt.plot(0,3.4, 'w', label=r'$porosity_{NE} = %0.3f$' %porosity_n)
        plt.plot(0,3.4, 'w', label=r'$porosity_{PE} = %0.3f$' %porosity_p)
        plt.plot(0,3.4, 'w', label=r'$C_{Rate} = %0.3f$' %C_rate)
        plt.legend(bbox_to_anchor=(1, 0.5))
        
    if abs(bc_p_OCV) < abs(Inputs_p[:, 2][0]):
        print('Out of bounds PE')
        
    if abs(bc_n_OCV) < abs(Inputs_n[:, 2][0]):
        print('Out of bounds NE')
    
    # return bc_n_OCV, bc_p_OCV