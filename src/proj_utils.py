from math import degrees, radians, atan, sin, cos, sqrt, tan

def scan_to_geod(y, x):
    r_eq = 6378137          # semi major axis of projection, m
    inv_f = 298.257222096   # inverse flattening
    r_pol = 6356752.31414   # semi minor axis of projection, m
    e = 0.0818191910435
    h_goes = 35786023       # perspective point height, m
    H = 42164160            # h_goes + r_eq, m
    lambda_0 = -1.308996939 # longitude of origin projection

    a = _calc_a(x, y, r_eq, r_pol)
    b = _calc_b(x, y, H)
    c = _calc_c(H, r_eq)
    r_s = _calc_rs(a, b, c)
    s_x = _calc_sx(r_s, x, y)
    s_y = _calc_sy(r_s, x)

    lat1 = (r_eq**2) / (r_pol**2)
    lat2 = s_z / (sqrt((H - s_x)**2 + s_y**2))
    lat = atan(lat1 * lat2)

    lon1 = atan(s_y / (H - s_x))
    lon = lambda_0 - lon1

    return (lat, lon)



def geod_to_scan(lat, lon):
    r_eq = 6378137          # semi major axis of projection, m
    inv_f = 298.257222096   # inverse flattening
    r_pol = 6356752.31414   # semi minor axis of projection, m
    e = 0.0818191910435
    h_goes = 35786023       # perspective point height, m
    H = 42164160            # h_goes + r_eq, m
    lambda_0 = -1.308996939 # longitude of origin projection

    lat = radians(lat)
    lon = radians(lon)

    theta_c = _calc_thetac(r_eq, r_pol, lat)
    r_c = _calc_rc(r_pol, e, theta_c)
    s_x = _calc_sx_inv(H, r_c, theta_c, lon, lambda_0)
    s_y = _calc_sy_inv(r_c, theta_c, lon, lambda_0)
    s_z = _calc_sz_inv(r_c, theta_c)

    y = atan(s_z / s_x)

    x = -s_y / (sqrt(s_x**2 + s_y**2 + s_z**2))

    return (y, x)



def _calc_a(x, y, r_eq, r_pol):
    f = sin(x)**2 + cos(x)**2
    g = cos(y)**2 + (r_eq**2 / r_pol**2) * (sin(y)**2)
    return f * g



def _calc_b(x, y, H):
    f = -2 * H * (cos(x)**2) * (cos(y)**2)
    return f



def _calc_c(H, r_eq):
    return (H**2 - r_eq**2)



def _calc_rs(a, b, c):
    num = -b - sqrt((b**2) - 4*a*c)
    den = 2 * a
    return num / den



def _calc_sx(r_s, x, y):
    s_x = r_s * cos(x) * cos(y)
    return s_x



def _calc_sy(r_s, x):
    s_y = -r_s * sin(x)
    return s_y



def _calc_sz(r_s, x, y):
    s_z = r_s * cos(x) * sin(y)



def _calc_thetac(r_eq, r_pol, lat):
    theta_c = (r_pol**2) / (r_eq**2)
    theta_c = theta_c * tan(lat)
    theta_c = atan(theta_c)
    return theta_c



def _calc_rc(r_pol, e, theta_c):
    den = sqrt(1 - e**2 * cos(theta_c)**2)
    r_c = r_pol / den
    return r_c



def _calc_sx_inv(H, r_c, theta_c, lon, lambda_0):
    s_x = H - (r_c * cos(theta_c) * cos(lon - lambda_0))
    return s_x



def _calc_sy_inv(r_c, theta_c, lon, lambda_0):
    s_y = -r_c * cos(theta_c) * sin(lon - lambda_0)
    return s_y



def _calc_sz_inv(r_c, theta_c):
    s_z = r_c * sin(theta_c)
    return s_z
