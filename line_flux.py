class LineFlux(object):
    def __init__(self, galaxy, lines=[]):
        self.galaxy = galaxy

    def set_lines(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class PlaceholderLineFlux(LineFlux):
    np.array([('OII_3726', 3726.032),
              ('OII_3729', 3728.815),
              ('NeIII_3869'3869.060),
              ('H_delta', 4101.734),
              ('H_gamma', 4340.464),
              ('H_beta', 4861.325),
              ('OIII_4959', 4958.911),
              ('OIII_5007', 5006.843),
              ('NII_6548', 6548.040),
              ('H_alpha', 6562.800),
              ('NII_6584', 6583.460),
              ('SII_6717', 6716.440),
              ('SII_6731', 6730.810)],
             dtype=[('names', 'S10'), ('wave', '<f8')])

    def __init__(self, galaxy, lines=[]):
        self.galaxy = galaxy

        mask = np.zeros(len(self.lines), dtype=bool)
        for line in lines:
        m = self.lines['name'] == line
        if not np.any(m):
            raise Exception('Line %s unknown to model' % line)
        mask += m

        self.lines = mask

    def __call__(self, SFR, Z):
        
        
        #SFR(Halpha) (M_sol/yr) = 7.9x10^-42 L(Halpha) (erg/s)
        flux = SFR / 7.9e-42 #erg/s
        var = flux / 10.
        return flux, var

