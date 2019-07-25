"""
21 June 2019
Author : Matt Nicholson

This file contains functions to convert GOES-16 N/S Elevation angle and E/W
Scanning Angle coordinates to GRS80 Ellipsoid latitude and longitude coordinates
and vice versa.

Equations obtained from:

United States, Congress, National Oceanic & Atmospheric Administration.
    “GOES R Series Product Definition and User's Guide.”
    GOES R Series Product Definition and User's Guide, 2nd ed., vol. 3,
    Harris Corporation, 2018, pp. 56–60.
"""

from math import degrees, radians, atan, sin, cos, sqrt, tan

def scan_to_geod(y, x):
    """
    Calculates the GRS80 ellipsoid geodetic latitude & longitude coordinates
    from GOES-R ABI fixed grid y & x coordinates

    Parameters
    ----------
    y : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians
    x : float
        GOES-R ABI fixed grid x coordinate representing the E/W Elevation Angle,
        in radians

    Returns
    -------
    tuple of floats
        Tuple containing two floats; the first is the Geodetic Latitude (in deg.),
        the second is the Geodetic Longitude (in deg.)
        Format: (lat, lon)
    """
    r_eq = 6378137          # semi major axis of projection, m
    inv_f = 298.257222096   # inverse flattening
    r_pol = 6356752.31414   # semi minor axis of projection, m
    e = 0.0818191910435
    h_goes = 35786023       # perspective point height, m
    H = 42164160            # h_goes + r_eq, m
    lambda_0 = -1.308996939 # longitude of origin projection

    if (not isinstance(x, float)):
        x = float(x)
    if (not isinstance(y, float)):
        y = float(y)

    a = _calc_a(x, y, r_eq, r_pol)
    b = _calc_b(x, y, H)
    c = _calc_c(H, r_eq)
    r_s = _calc_rs(a, b, c)
    s_x = _calc_sx(r_s, x, y)
    s_y = _calc_sy(r_s, x)
    s_z = _calc_sz(r_s, x, y)

    lat1 = (r_eq**2) / (r_pol**2)
    lat2 = s_z / (sqrt((H - s_x)**2 + s_y**2))
    lat = atan(lat1 * lat2)

    lon1 = atan(s_y / (H - s_x))
    lon = lambda_0 - lon1

    ################### For debugging ###################
    # print('a ', a)
    # print('b ', b)
    # print('c ', c)
    # print('r_s ', r_s)
    # print('s_x ', s_x)
    # print('s_y ', s_y)
    # print('s_z ', s_z)
    #####################################################

    lon = degrees(lon)
    lat = degrees(lat)

    return (lat, lon)



def geod_to_scan(lat, lon):
    """
    Calculates GOES-R ABI fixed grid y & x coordinates from latitude and longitude
    coordinates

    Parameters
    ----------
    lat : float
        GRS80 Ellipsoid geodetic latitude coordinate, in degrees
    lon : float
        GRS80 Ellipsoid geodetic longitude coordinate, in degrees

    Returns
    -------
    tuple of floats
        Tuple of two floats; the first is the N/S Elevation angle (in radians),
        the second is the E/W Elevation angle (in radians)
    """
    r_eq = 6378137          # semi major axis of projection, m
    inv_f = 298.257222096   # inverse flattening
    r_pol = 6356752.31414   # semi minor axis of projection, m
    e = 0.0818191910435
    h_goes = 35786023       # perspective point height, m
    H = 42164160            # h_goes + r_eq, m
    lambda_0 = -1.308996939 # longitude of origin projection

    if (not isinstance(lat, float)):
        lat = float(lat)
    if (not isinstance(lon, float)):
        lon = float(lon)

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
    """
    Calculates the a coefficient used to calculate the distance from the satellite
    to a point P

    Parameters
    ----------
    x : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians
    y : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians
    r_eq : int
        GOES-16 semi-major axis of projection, in meters
    r_pol
        GOES-16 semi-minor axis of projection, in meters

    Returns
    -------
    float
    """
    f = sin(x)**2 + cos(x)**2
    g = cos(y)**2 + (r_eq**2 / r_pol**2) * (sin(y)**2)
    return f * g



def _calc_b(x, y, H):
    """
    Calculates the b coefficient used to calculate the distance from the satellite
    to a point P

    Parameters
    ----------
    x : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians
    y : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians
    H : int
        GOES-16 projection perspective point height, in meters

    Returns
    -------
    float
    """
    f = -2 * H * (cos(x)) * (cos(y))
    return f



def _calc_c(H, r_eq):
    """
    Calculates the b coefficient used to calculate the distance from the satellite
    to a point P

    Parameters
    ----------
    H : int
        GOES-16 projection perspective point height, in meters
    r_eq : int
        GOES-16 semi-major axis of projection, in meters

    Returns
    -------
    float
    """
    return (H**2 - r_eq**2)



def _calc_rs(a, b, c):
    """
    Calculates the distance from the satellite to a point P using the quadratic
    formula

    Parameters
    ----------
    a : float
        'a' coefficient for the quadratic formula
    b : float
        b coefficient for the quadratic formula
    c : float
        c coefficient for the quadratic formula

    Returns
    -------
    float
        distance of the satellite from point P, in meters
    """
    try:
        num = -b - sqrt((b**2) - 4*a*c)
    except ValueError:
        print('\nWarning: ValueError caught in _calc_rs')
        print('Attempting again with the absolute value\n')
        num = -b - sqrt(abs((b**2) - 4*a*c))
    den = 2 * a
    return num / den



def _calc_sx(r_s, x, y):
    """
    Calculates the x-axis of the satellite coordinate frame for scan_to_geod

    Parameters
    ----------
    r_s : float
        Distance from the satellite to point P, in meters
    x : float
        GOES-R ABI fixed grid x coordinate representing the E/W Elevation Angle,
        in radians
    y : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians

    Returns
    -------
    s_x : float
        x-axis of the satellite coordinate frame
    """
    s_x = r_s * cos(x) * cos(y)
    return s_x



def _calc_sy(r_s, x):
    """
    Calculates the y-axis of the satellite coordinate frame for scan_to_geod

    Parameters
    ----------
    r_s : float
        Distance from the satellite to point P, in meters
    x : float
        GOES-R ABI fixed grid x coordinate representing the E/W Elevation Angle,
        in radians

    Returns
    -------
    s_y : float
        y-axis of the satellite coordinate frame
    """
    s_y = -r_s * sin(x)
    return s_y



def _calc_sz(r_s, x, y):
    """
    Calculates the z-axis of the satellite coordinate frame for scan_to_geod

    Parameters
    ----------
    r_s : float
        Distance from the satellite to point P, in meters
    x : float
        GOES-R ABI fixed grid x coordinate representing the E/W Elevation Angle,
        in radians
    y : float
        GOES-R ABI fixed grid y coordinate representing the N/S Elevation Angle,
        in radians

    Returns
    -------
    s_z : float
        z-axis of the satellite coordinate frame
    """
    s_z = r_s * cos(x) * sin(y)
    return s_z



def _calc_thetac(r_eq, r_pol, lat):
    """
    Calculates the geocentric latitude, in radians

    Parameters
    ----------
    r_eq : int
        GOES-16 semi-major axis of projection, in meters
    r_pol : int
        GOES-16 semi-minor axis of projection, in meters
    lat : float
        GRS80 geodetic latitude, in radians

    Returns
    -------
    theta_c : float
        Geocentric latitude, in radians
    """
    theta_c = (r_pol**2) / (r_eq**2)
    theta_c = theta_c * tan(lat)
    theta_c = atan(theta_c)
    return theta_c



def _calc_rc(r_pol, e, theta_c):
    """
    Calculates the geodetic distance to the point on the ellipsoid

    Parameters
    ----------
    r_pol : int
        GOES-16 semi-minor axis of projection, in meters
    e : float
        Uh...
    theta_c
        Geocentric latitude, in radians

    Returns
    -------
    r_c : float
        Geocentric distance to the point on the ellipsoid, in meters
    """
    den = sqrt(1 - e**2 * cos(theta_c)**2)
    r_c = r_pol / den
    return r_c



def _calc_sx_inv(H, r_c, theta_c, lon, lambda_0):
    """
    Calculates the x-axis of the satellite coordinate frame for geod_to_scan

    Parameters
    ----------
    H : int
        GOES-16 imagery projection perspective point height + semi-major axis, in meters
    r_c : float
        Geodetic distance to the point on the ellipsoid
    theta_c : float
        Geocentric latitude, in radians
    lon : float
        Geodetic longitude coordinate, in radians
    lambda_0: float
        GOES-16 longitude of projection origin, in radians

    Returns
    -------
    s_x : float
        X-axis of the satellite coordinate frame
    """
    s_x = H - (r_c * cos(theta_c) * cos(lon - lambda_0))
    return s_x



def _calc_sy_inv(r_c, theta_c, lon, lambda_0):
    """
    Calculates the y-axis of the satellite coordinate frame for geod_to_scan

    Parameters
    ----------
    r_c : float
        Geodetic distance to the point on the ellipsoid
    theta_c : float
        Geocentric latitude, in radians
    lon : float
        Geodetic longitude coordinate, in radians
    lambda_0: float
        GOES-16 longitude of projection origin, in radians

    Returns
    -------
    s_y : float
        Y-axis of the satellite coordinate frame
    """
    s_y = -r_c * cos(theta_c) * sin(lon - lambda_0)
    return s_y



def _calc_sz_inv(r_c, theta_c):
    """
    Calculates the z-axis of the satellite coordinate frame for geod_to_scan

    Parameters
    ----------
    r_c : float
        Geodetic distance to the point on the ellipsoid
    theta_c : float
        Geocentric latitude, in radians

    Returns
    -------
    s_z : float
        Z-axis of the satellite coordinate frame
    """
    s_z = r_c * sin(theta_c)
    return s_z




# print(scan_to_geod(0.095340, -0.024052))
# print(geod_to_scan(48.0563, -70.1242)) # (0.12390027294128181, 0.00950246171751412)
# print(geod_to_scan(33.943546, -84.52599)) # (0.09557181243071429, -0.023614853364253407)
# print(geod_to_scan(33.8461, -84.6909)) # (0.09533985706770494, -0.024049622722219145)
# print('NW: ', scan_to_geod(0.11088, -0.0728))
# print('NE: ', scan_to_geod(0.11088, -0.0448))
# print('SW: ', scan_to_geod(0.08288, -0.0728))
# print('SE: ', scan_to_geod(0.08288, -0.0448))
