import numpy as np
class aia_psf():

    def __init__(self, wavelength, use_preflightcore=False, qabort=False,
                        npix = 4096, dwavelength = 0, dpx = 0.5, dpy = 0.5):

        self.use_preflightcore = use_preflightcore
        self.dwavelength = dwavelength
        self.wavelength = wavelength
        self.npix = npix
        self.dpx = dpx
        self.dpy = dpy

    @property
    def ss_wave(self):
        ss_wave = { '94':'94', '131':'131', '171':'171', '193':'193',\
                    '211':'211', '304':'304', '335':'335', }
        return ss_wave[self.wavelength]

    @property
    def channel(self):
        chanel = { '94':94, '131':131, '171':171, '193':193, \
                   '211':211, '304':304, '335':335 ,}
        return channel[self.wavelength]

    @property
    def image(self):
        image = { '94':'AIA20101016_191039_0094.fits',
                  '131':'AIA20101016_191035_0131.fits',
                  '171':'AIA20101016_191037_0171.fits',
                  '193':'AIA20101016_191056_0193.fits',
                  '211':'AIA20101016_191038_0211.fits',
                  '304':'AIA20101016_191021_0304.fits',
                  '335':'AIA20101016_191041_0335.fits'}
        return image[self.wavelength]

    @property
    def refimage(self):
        refimage = { '94':'AIA20101016_190903_0094.fits',
                     '131':'AIA20101016_190911_0131.fits',
                     '171':'AIA20101016_190901_0171.fits',
                     '193':'AIA20101016_190844_0193.fits',
                     '211':'AIA20101016_190902_0211.fits',
                     '304':'AIA20101016_190845_0304.fits',
                     '335':'AIA20101016_190905_0335.fits',}
        return refimage[self.wavelength]

    @property
    def angle1(self):
        angle1 = { '94':49.81, '131':50.27 , '171':49.81 , '193':49.82,
                    '211':49.78, '304':49.76 , '335':50.40}
        return angle1[self.wavelength]

    @property
    def angle2(self):
        angle2 = { '94':40.16, '131':40.17 , '171':39.57 , '193':39.57,
                                '211':40.08, '304':40.18 , '335':39.80}
        return angle2[self.wavelength]

    @property
    def angle3(self):
        angle3 = { '94':-40.28, '131':-39.70 , '171':-40.13 , '193':-40.12,
                                '211':-40.34, '304':-40.14 , '335':-39.64}
        return angle3[self.wavelength]

    @property
    def angle4(self):
        angle4 = { '94':-49.92, '131':-49.95 , '171':-50.38, '193':-50.37,
                                '211':-49.95, '304':-49.90 , '335':-50.25}
        return angle4[self.wavelength]

    @property
    def delta1(self):
        delta1 = { '94':0.02, '131':0.02, '171':0.02, '193':0.02,
                                '211':0.02, '304':0.02, '335':0.02}
        return delta1[self.wavelenght]

    @property
    def delta2(self):
        delta2 = { '94':0.02, '131':0.02, '171':0.02, '193':0.02,
                                '211':0.02, '304':0.02, '335':0.02}
        return delta2[self.wavelenght]

    @property
    def delta3(self):
        delta3 = { '94':0.02, '131':0.02, '171':0.02, '193':0.02,
                                '211':0.02, '304':0.02, '335':0.02}
        return delta3[self.wavelenght]

    @property
    def delta4(self):
        delta4 = { '94':0.02, '131':0.02, '171':0.02, '193':0.02,
                                '211':0.02, '304':0.02, '335':0.02}
        return delta4[self.wavelenght]

    @property
    def spacing(self):
        spacing = { '94':8.99, '131':12.37 , '171':16.26, '193':18.39,
                                '211':19.97, '304':28.87 , '335':31.83}
        return spacing[self.wavelength] * (1.0 + self.dwavelength)

    @property
    def dspacing(self):
        dspacing = { '94':0.13, '131':0.16 , '171':0.1, '193':0.20,
                                '211':0.09, '304':0.05 , '335':0.07}
        return dspacing[self.wavelength]

    @property
    def solarnorth(self):
        solarnorth = { '94':'UP', '131':'UP' , '171':'UP', '193':'UP',
                                '211':'UP', '304':'UP', '335':'UP'}
        return solarnorth[self.wavelength]

    @property
    def meshpitch(self):
        meshpitch = { '94':363.0, '131':363.0 , '171':363.0, '193':363.0,
                                '211':363.0, '304':363.0 , '335':363.0}
        return meshpitch[self.wavelength]

    @property
    def meshwidth(self):
        meshwidth = { '94':34.0, '131':34.0, '171':34.0, '193':34.0,
                                '211':34.0, '304':34.0 , '335':34.0}
        return meshwidth[self.wavelength]

    @property
    def gs_width(self):
        if self.use_preflightcore:
            gs_width = { '94':4.5, '131':4.5, '171':4.5, '193':4.5,
                                '211':4.5, '304':4.5 , '335':4.5}
        else:
            gs_width = { '94':0.951, '131':1.033, '171':0.962, '193':1.512,
                                '211':1.199, '304':1.247, '335':0.962}
        return gs_width[self.wavelength]

    @property
    def fp_spacing(self):
        fp_spacing = { '94':0.207, '131':0.289, '171':0.377, '193':0.425,
                       '211':0.465, '304':0.670 , '335':0.738}
        return fp_spacing[self.wavelength]


    def generate_gaussian_core(self, I, xc, yc ):

        x = np.arange(0.,self.npix)+0.5
        gx = np.exp(-self.gs_width*(x-xc)**2)
        gy = np.exp(-self.gs_width*(x-yc)**2)

        return np.outer(gx, gy) * I


    def get_entrance_filter( self ):

        import math

        dx1 = self.spacing * np.cos(self.angle1 / 180. * math.pi)
        dy1 = self.spacing * np.sin(self.angle1 / 180. * math.pi)

        dx2 = self.spacing * np.cos(self.angle2 / 180. * math.pi)
        dy2 = self.spacing * np.sin(self.angle2 / 180. * math.pi)

        dx3 = self.spacing * np.cos(self.angle3 / 180. * math.pi)
        dy3 = self.spacing * np.sin(self.angle3 / 180. * math.pi)

        dx4 = self.spacing * np.cos(self.angle4 / 180. * math.pi)
        dy4 = self.spacing * np.sin(self.angle4 / 180. * math.pi)

        x = np.arange(0.,self.npix)+0.5

        psf = np.zeros((self.npix,self.npix))

        for j in np.arange(-100,100):
            if j != 0:

                I0 = np.sinc( j * self.meshwidth / self.meshpitch )**2

                xc = (self.npix/2.0) + (dx1 * j) + self.dpx
                yc = (self.npix/2.0) + (dy1 * j) + self.dpy
                psf += self.generate_gaussian_core(I0, xc, yc )

                xc = (self.npix/2.0) + (dx2 * j) + self.dpx
                yc = (self.npix/2.0) + (dy2 * j) + self.dpy
                psf += self.generate_gaussian_core(I0, xc, yc )

                xc = (self.npix/2.0) + (dx3 * j) + self.dpx
                yc = (self.npix/2.0) + (dy3 * j) + self.dpy
                psf += self.generate_gaussian_core(I0, xc, yc )

                xc = (self.npix/2.0) + (dx4 * j) + self.dpx
                yc = (self.npix/2.0) + (dy4 * j) + self.dpy
                psf += self.generate_gaussian_core(I0, xc, yc )

        xc = (self.npix/2.0) + self.dpx
        yc = (self.npix/2.0) + self.dpy
        psf2 = self.generate_gaussian_core(1, xc, yc)

        return (psf / np.sum(psf) * 0.18) + (psf2 / np.sum(psf2) * 0.82)


    def get_focal_plane( self ):

        import math

        psf=np.zeros((self.npix,self.npix))

        dx1 = self.spacing * np.cos( math.pi / 4 )
        dy1 = self.spacing * np.sin( math.pi / 4 )
        dx2 = self.spacing * np.cos(-math.pi / 4 )
        dy2 = self.spacing * np.sin(-math.pi / 4 )

        for j in np.arange(-100,100):

            I0 = np.sinc(j * self.meshwidth / self.meshpitch)**2

            xc = (self.npix/2.0) + (dx1*j) + self.dpx
            yc = (self.npix/2.0) + (dy1*j) + self.dpy
            psf += self.generate_gaussian_core(I0, xc, yc )

            xc = (self.npix/2.0) + (dx2*j) + self.dpx
            yc = (self.npix/2.0) + (dy2*j) + self.dpy
            psf += self.generate_gaussian_core(I0, xc, yc)

            xc = (self.npix/2.0) - (dx1*j) + self.dpx
            yc = (self.npix/2.0) - (dy1*j) + self.dpy
            psf += self.generate_gaussian_core(I0, xc, yc)

            xc = (self.npix/2.0) - (dx2*j) + self.dpx
            yc = (self.npix/2.0) - (dy2*j) + self.dpy
            psf += self.generate_gaussian_core(I0, xc, yc)

        xc = (self.npix/2.0) + self.dpx
        yc = (self.npix/2.0) + self.dpy
        psf2 = self.generate_gaussian_core(1, xc, yc)

        #return (psf / np.sum(psf) * 0.18) + (psf2 / np.sum(psf2) * 0.82)
        return (psf2 / np.sum(psf2) * 0.82)


    def aia_diffractionpattern( self ):

        from scipy.signal import fftconvolve

        PSFEntranceFilter = self.get_entrance_filter()
        PSFFocalPlane = self.get_focal_plane()

        OTFFocalPlane = np.fft.fft2(PSFFocalPlane)
        OTFEntranceFilter = np.fft.fft2(PSFEntranceFilter)

        PSFComplete = (np.fft.ifft2(OTFFocalPlane * OTFEntranceFilter))
        PSFComplete = np.abs(np.fft.ifftshift(PSFComplete))

        return PSFComplete / np.sum(PSFComplete)


    def get( self ):
        return self.aia_diffractionpattern( ).clip(min=0)
