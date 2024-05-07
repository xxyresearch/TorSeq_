import numpy as np

def get_corr_of_cos_i_and_sin_j(data):
    M = data.shape[1]
    cos_data = np.cos(data)
    sin_data = np.sin(data)
    corr_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            corr_matrix[i, j] = np.corrcoef(cos_data[:, i], sin_data[:, j])[0, 1]
    
    return corr_matrix

def get_corr_of_cos_j_and_sin_i(data):
    cos_data = np.cos(data)
    sin_data = np.sin(data)
    M = data.shape[1]
    corr_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            corr_matrix[i, j] = np.corrcoef(sin_data[:, i], cos_data[:, j])[0, 1]
    
    return corr_matrix

def cosine_sine_same_angle_correlation(data):
    cos_data = np.cos(data)
    sin_data = np.sin(data)
    M = data.shape[1]
    correlations = []
    for i in range(M):
        correlations.append(np.corrcoef(cos_data[:, i], sin_data[:, i])[0, 1])
    return np.tile(correlations, (len(correlations),1)).transpose()

def get_squared_canonical_correlation_coefficient(data):
        r_cc = np.corrcoef(np.cos(data), rowvar=False)
        r_cs = get_corr_of_cos_i_and_sin_j(data)
        r_ss = np.corrcoef(np.sin(data), rowvar=False)
        r_sc = get_corr_of_cos_j_and_sin_i(data)
        r_1 = cosine_sine_same_angle_correlation(data)
        r_2 = r_1.transpose()
        r_ones = np.ones_like(r_1)
        sqr_r = ((r_cc ** 2 + r_cs ** 2 + r_ss ** 2 + r_sc ** 2)  
                + 2 * (r_cc * r_ss + r_cs * r_sc) * r_1 * r_2 
                - 2 * (r_cc * r_cs + r_sc * r_ss) * r_2  
                - 2 * (r_cc * r_sc + r_cs * r_ss) * r_1)/ ((r_ones- r_1 ** 2) * ( r_ones- r_2 ** 2))
        return sqr_r


