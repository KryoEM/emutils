import numpy as np
import image

def kev_to_lam(vol):
    return 12.26 / np.sqrt(1000.0 * vol + 0.9784 * pow(1000.0 * vol, 2) / pow(10.0, 6))

def xy_to_az(x,y):
    lzloc = np.logical_or(x <= 0,y <= 0)
    res   = np.arctan2(y,x)
    res[lzloc] = 0
    return  res

class CTF:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.lam    = kev_to_lam(float(self.Voltage))
        A           = float(self.AmplitudeContrast)
        self.A_term = np.arctan(A/np.sqrt(1.0-A))
        if not 'PhaseShift' in kwargs:
            self.PhaseShift = 0.0
        #precomputed_amplitude_contrast_term = atan(amplitude_contrast / sqrt(1.0 - amplitude_contrast));

    def ctf_2d(self,sz,psize):
        # switch to python "C" convention
        y, x = image.cart_coords2D(sz)
        x, y = x / sz[1], y / sz[0]
        r    = np.sqrt(x ** 2 + y ** 2)
        s    = r / psize
        az   = xy_to_az(x,y)
        c    = self.evaluate(s**2,az)
        return np.fft.ifftshift(c)

    def evaluate(self,s2,az):
        ''' Return the value of the CTF at the given squared spatial frequency and azimuth '''
        return -np.sin(self.s2_az_to_phase(s2,az))

    def az_to_defocus(self,az):
        '''Return the effective defocus at the azimuth of interest'''
        defocus1 = float(self.DefocusU)
        defocus2 = float(self.DefocusV)
        astig_az = float(self.DefocusAngle)
        return 0.5*(defocus1 + defocus2 + np.cos(2.0*(astig_az-az))*(defocus1-defocus2))
        #return 0.5 * (defocus_1 + defocus_2 + cos(2.0 * (azimuth - astigmatism_azimuth)) * (defocus_1 - defocus_2));

    def s2_az_to_phase(self,s2,az):
        ''' Returns the argument (radians) to the sine and cosine terms of the ctf
            We follow the convention, like the rest of the cryo-EM/3DEM field, that underfocusing the objective lens
            gives rise to a positive phase shift of scattered electrons, whereas the spherical aberration gives a
            negative phase shift of scattered electrons.
            Note that there is an additional (precomputed) term so that the CTF can then be computed by simply
            taking the sine of the returned phase shift. '''
        assert(np.all(s2) >=0 )
        lam = self.lam
        cs  = float(self.SphericalAberration)
        phase_shift = np.mod(float(self.PhaseShift),np.pi)
        return np.pi*lam*s2*(self.az_to_defocus(az) - 0.5*(lam**2)*s2*cs) + phase_shift + self.A_term
        # return PI * wavelength * squared_spatial_frequency * (DefocusGivenAzimuth(
        #     azimuth) - 0.5 * squared_wavelength * squared_spatial_frequency * spherical_aberration) + additional_phase_shift + precomputed_amplitude_contrast_term;
        # }

    def phase_flip_cpu(self,im,psize):
        c    = self.ctf_2d(im.shape,psize)
        imff = np.fft.ifftn(np.fft.fftn(im)*np.sign(c))
        return np.float32(np.real(imff))

