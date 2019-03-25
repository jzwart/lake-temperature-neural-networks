# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:19:10 2018

@author: aappling
"""

import numpy as np
import tensorflow as tf

def transformTempToDensity(temp):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    densities = 1000*(1-((temp+288.9414)*tf.pow(temp- 3.9863,2))/(508929.2*(temp+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))

    return densities

#def calculate_density_loss(prd, npic, n_steps):
#                # calculate phy-loss
#                ploss = 0
#                prd_d = 1000*(1-(prd+288.9414)*((prd-3.9863)**2)/(508929.2*(prd+68.12963))) # density
#                for k in range(51*npic-1):
#                    if (k+1)%51!=0:
#                        dif = np.reshape((prd_d[k,:,:]-prd_d[k+1,:,:]>0),[1,n_steps])
#                        dif = np.maximum(np.zeros([1,n_steps]),dif)
#                        ploss = ploss+np.sum(dif)
#                ploss = ploss/n_steps/npic/50
#                return ploss

def calculate_ec_loss(inputs, outputs, phys, depth_areas, n_depths, ec_threshold, n_sets, colnames_physics, combine_days=1):
    #******************************************************
    #description: calculates energy conservation loss
    #parameters:
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@n_sets: number of sets of depths in the batch
        #@colnames_physics: numpy array of the column names for the feature dimension of the physics array
        #@combine_days: how many days to look back to see if energy is conserved
    #*********************************************************************************

#    diff_vec = torch.empty((inputs.size()[1]))
#    n_dates = inputs.size()[1]
    # outputs = labels

#    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs)
    # print("modeled densities: ", densities)

    diff_per_set = [] #torch.empty(n_sets)
    for i in range(n_sets):
        #loop through sets of n_depths
        #indices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths

        #calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], n_depths, depth_areas)

        #calculate energy change in each timestep
        surface_area = depth_areas[0,1]
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, surface_area)
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index,:,:], outputs[start_index,:], combine_days, colnames_physics)
#        ### can use this to plot energy delta and flux over time to see if they line up
#        doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
#        doy = doy[1:-1]

#        print(lake_energy_deltas)
        diff_vec = tf.abs(lake_energy_deltas - lake_energy_fluxes) #.abs_()

        # mendota og ice guesstimate
        # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

        # don't apply EC to the ice-on period. the mask is a column in the phys matrix
        # (it's a boolean, i.e., 0 for ice, 1 for non-ice)
        ice_col = phys_column_index('Ice', colnames_physics)
        tmp_mask = 1-phys[start_index+1,1:-1,ice_col]
        tmp_loss = tf.reduce_mean(diff_vec*tf.cast(tmp_mask,tf.float32))
        diff_per_set.append(tmp_loss)

##        print(phys)
#        diff_vec = diff_vec[tf.where((phys[1:(n_depths-tf.shape(diff_vec)[0]-1),9] == 0))[0]]
#        # #compute difference to be used as penalty
#        if tf.shape(diff_vec)[0]==0: #.size() == torch.Size([0]):
#            diff_per_set.append(0) #diff_per_set[i] = 0
#        else:
#            diff_per_set.append(tf.reduce_mean(diff_vec)) #diff_per_set[i] = diff_vec.mean()

    diff_per_set_r = tf.stack(diff_per_set)

    diff_per_set = tf.clip_by_value(diff_per_set_r - ec_threshold, clip_value_min=0,clip_value_max=999999)
#    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    return tf.reduce_mean(diff_per_set),diff_vec,diff_per_set_r,diff_per_set #, lake_energy_deltas, lake_energy_fluxes#.mean()

def calculate_lake_energy(temps, densities, n_depths, depth_areas):
    #calculate the total energy of the lake for every timestep
    #sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    #then multiply by the specific heat of water
    dz = 0.5 #thickness for each layer, hardcoded for now
    cw = 4186 #specific heat of water

    areas = tf.reshape(depth_areas[:,1],[n_depths,1])
    energy = tf.reduce_sum(tf.multiply(tf.cast(areas,tf.float32),temps)*densities*dz*cw,0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    #given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
#    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400 #seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1])/time/surface_area
#    energy_deltas = (energies[1:] - energies[:-1])/(time*surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas


def calculate_air_density(air_temp, rh):
    #returns air density in kg / m^3
    #equation from page 13 GLM/GLEON paper(et al Hipsey)

    #Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    #atmospheric pressure
    p = 1013. #mb

    #water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh,air_temp)

    #water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure/(p - vapPressure)
    # print( 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15)))
    # print("vs")
    # print(1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))
    # sys.exit()
    # return 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15))
    return (1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))*100#
def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 22 in GLM/GLEON paper(et al Hipsey)
    #GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    #calculate air density
    rho_a = calculate_air_density(air_temp, rel_hum)

    #specific heat capacity of air in J/(kg*C)
    c_a = 1005.


    #bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    #wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    # U_10 = wind_speed
    return -rho_a*c_a*c_H*U_10*(surf_temp - air_temp)

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 23 in GLM/GLEON paper(et al Hipsey)
    #GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    #air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    #bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    #latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    #wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
#
    #ratio of molecular weight of water to that of dry air
    omega = 0.622

    #air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a*c_E*lambda_v*U_10*(omega/p)*(e_s - e_a)


def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)

def calculate_vapour_pressure_saturated(temp):
    # returns in miilibars
    # print(torch.pow(10, (9.28603523 - (2332.37885/(temp+273.15)))))

    #Converted pow function to exp function workaround pytorch not having autograd implemented for pow
    exponent = (9.28603523 - (2332.37885/(temp+273.15))*np.log(10))
    return tf.exp(exponent)

def calculate_wind_speed_10m(ws, ref_height=2.):
    #from GLM code glm_surface.c
    c_z0 = 0.001 #default roughness
    return ws*(tf.log(10.0/c_z0)/tf.log(ref_height/c_z0))


def calculate_energy_fluxes(phys, surf_temps, combine_days, colnames_physics):

    short_wave_col = phys_column_index('ShortWave', colnames_physics)
    R_sw_arr = phys[:-1,short_wave_col] + (phys[1:,short_wave_col]-phys[:-1,short_wave_col])/2

    long_wave_col = phys_column_index('LongWave', colnames_physics)
    R_lw_arr = phys[:-1,long_wave_col] + (phys[1:,long_wave_col]-phys[:-1,long_wave_col])/2

    e_s = 0.985 #emissivity of water, given by Jordan
    sigma = 5.67e-8 #Stefan-Boltzmann constant
    R_lw_out_arr = e_s*sigma*(tf.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]

    air_temp_col = phys_column_index('AirTemp', colnames_physics)
    air_temp = phys[:-1,air_temp_col]
    air_temp2 = phys[1:,air_temp_col]

    rel_hum_col = phys_column_index('RelHum', colnames_physics)
    rel_hum = phys[:-1,rel_hum_col]
    rel_hum2 = phys[1:,rel_hum_col]

    wind_speed_col = phys_column_index('WindSpeed', colnames_physics)
    ws = phys[:-1, wind_speed_col]
    ws2 = phys[1:,wind_speed_col]

    E1 = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H1 = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E1 + E2)/2
    H = (H1 + H2)/2

    # Combine into net flux
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albedo, given by Jordan Read
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])

    return fluxes

def phys_column_index(colname, colnames_physics):
    return [i for i, v in enumerate(colnames_physics) if colname in v][0]
