class MetallicityProfile(object):
    def required_params(self):
        raise NotImplementedError
    def __call__(self):
        raise NotImplementedError

class LinearMetallicityProfile(MetallicityProfile):

    def required_params(self):
        params = ('r0', 'Z_in', 'Z_out')
        return params

    def __call__(self, r, params):
        r0 = params['r0'])
        Z_in = params['Z_in']
        Z_out = params['Z_out']
        Z = (Z_out - Z_in) * (r / r0) + Z_in
        return Z
