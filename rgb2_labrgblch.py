import numpy as np
import pandas as pd


def cartorgb2rgb (cartorgb):
    #rgbslabch contains
        depth=float(255.0)

        rgbslabch = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0])
##        s = (cartorgb/pow(2,24))
##        b = ((cartorgb - np.floor(s)*pow(2,24))/pow(2,16))
##        g = ((cartorgb - np.floor(s)*pow(2,24) - np.floor(b)*pow(2,16))/pow(2,8))
##        r = (cartorgb - np.floor(s)*pow(2,24) - np.floor(b)*pow(2,16) -np.floor(g)*pow(2,8))
        s = np.floor(cartorgb/pow(2,24))
        b = np.floor((cartorgb - s*pow(2,24))/pow(2,16))
        g = np.floor((cartorgb - s*pow(2,24) - b*pow(2,16))/pow(2,8))
        r = np.floor(cartorgb - s*pow(2,24) - b*pow(2,16) -g*pow(2,8))
        #s=r
        #print(s, r, g, s*np.float_power(2.0,24.0), cartorgb1)
##        print(s, r, g, b, s)
        rgbslabch[0] =r
        rgbslabch[1]=g
        rgbslabch[2]=b
        rgbslabch[3]=s
        rgbslabch[4] =r/depth
        rgbslabch[5]=g/depth
        rgbslabch[6]=b/depth
        
        lab=rgb2lab(rgbslabch[4]/s,rgbslabch[5]/s,rgbslabch[6]/s,1)
        #lab=rgb2lab2(np.array([r/s,g/s,b/s]))
        #lab=rgb2lab(r/s,g/s,b/s,255)
        rgbslabch[7]=lab[0]
        rgbslabch[8]=lab[1]
        rgbslabch[9]=lab[2]
        rgbslabch[10]=np.sqrt(lab[1]*lab[1]+lab[2]*lab[2])
        rgbslabch[11]=np.mod(180*np.arctan2(lab[2],lab[1])/np.pi +360, 360)
        #print(rgbslabch)
        return rgbslabch



##############another example from web gave same results###################
# RGB to Lab conversion

# Step 1: RGB to XYZ
#         http://www.easyrgb.com/index.php?X=MATH&H=02#text2
# Step 2: XYZ to Lab
#         http://www.easyrgb.com/index.php?X=MATH&H=07#text7


def rgb2lab2(inputColor):

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab
###############################        
def rgb2lab(red,green,blue,depth,illuminant='D65'):

    rgb = np.array([red/depth,green/depth,blue/depth])
    for i,j in enumerate(rgb):
        if j > 0.04045 :
            rgb[i] = ( ( j + 0.055 ) / 1.055 ) ** 2.4
        else :
            rgb[i] = j / 12.92

    # Convert to XYZ. Need to decide what RGB we are using
    # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # https://en.wikipedia.org/wiki/CIELAB_color_space

    if illuminant == 'D50':
        # sRGB
        M = np.array([[0.4360747,  0.3850649,  0.1430804], \
                     [0.2225045,  0.7168786,  0.0606169], \
                     [0.0139322,  0.0971045,  0.7141733]])
        Xn = 96.4242
        Yn = 100.
        Zn = 82.5188

    elif illuminant == 'D65':
        M = np.array([[0.4124564,  0.3575761,  0.1804375], \
                      [0.2126729,  0.7151522,  0.0721750], \
                      [0.0193339,  0.1191920,  0.9503041]])
        # D65 illuminant
        Xn = 95.0489
        Yn = 100.
        Zn = 108.5188

    else:
        error("currently supported illuminants are 'D50' or 'D65'")

    XYZ = np.dot(M,rgb)*100

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    # Convert to CIE LAB
    # https://en.wikipedia.org/wiki/CIELAB_color_space
    # CIE XYZ tristimulus for chosen illuminant
    def f(t):
        delta = 6./29
        if (t>delta**3):
            value = t**(1/3)
        else:
            value = t/(3*delta**2) + 4/29
        return value

    # Finally convert to Lab
    L = 116*f(Y/Yn) - 16
    a = 500*(f(X/Xn)-f(Y/Yn))
    b = 200*(f(Y/Yn)-f(Z/Zn))

    return (L,a,b)

def lab2rgb(L,a,b,illuminant='D50'):

    if illuminant == 'D50':
        # sRGB
        invM = np.array([[3.1338561, -1.6168667, -0.4906146], \
                         [-0.9787684,  1.9161415,  0.0334540], \
                         [0.0719453, -0.2289914,  1.4052427]])
        Xn = 96.4242
        Yn = 100.
        Zn = 82.5188

    elif illuminant == 'D65':
        invM = np.array([[3.2404542, -1.5371385, -0.4985314], \
                         [-0.9692660,  1.8760108,  0.0415560], \
                         [0.0556434, -0.2040259,  1.0572252]])

        # D65 illuminant
        Xn = 95.0489
        Yn = 100.
        Zn = 108.5188

    def finv(t):
        delta = 6./29
        if (t>delta):
            value = t**3
        else:
            value = (3*delta**2)*(t - 4/29)
        return value

df = pd.read_csv("carto_colour.csv", usecols = ['colour'])
output = pd.DataFrame([], columns=['R-norm','G-norm','B-norm','S','R', 'G','B','L','a','b','C','H'])
for i in range(len(df)):
    #print(df.loc[i].at["colour"])
    out=cartorgb2rgb(df.loc[i].at["colour"])
    output.loc[len(output)] = out
    #print(out)
#print(output)
output.to_csv('RGBSLabCH.csv')
#print(df)
