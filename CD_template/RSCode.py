import galois
import numpy as np

class RSCode:
    def __init__(self, m,t,l,m0):
        self.m = m #GF(2^m) field
        self.t = t #Error correction capability
        self.n = 2**m-1 #Code length
        self.k = self.n-2*t #Information length
        self.l = l #Shortened information length (-> shortened code length = l+n-k)
        self.m0 = m0 #m0 of the Reed-Solomon code, determines first root of generator

        self.g = self.makeGenerator(m,t,m0) # generator polynomial represented by a galois.Poly variable

    def encode(self,msg):
        # Systematically encodes information words using the Reed-Solomon code
        # Input:
        #  -msg: a 2D array of galois.GF elements, every row corresponds with a GF(2^m) information word of length self.l
        # Output:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword corresponding to systematic Reed-Solomon coding of the corresponding information word
        assert np.shape(msg)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(msg) is galois.GF(2**self.m) , 'each element of msg  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        n_words = np.shape(msg)[0]
        n_parity = self.n - self.k  # = 2t parity symbols
        cw_len = self.l + n_parity

        code = GF(np.zeros((n_words, cw_len), dtype=int))

        # Systematic encoding: compute remainder of (msg(x) * x^{2t}) / g(x).
        # The codeword is [message | parity], so message occupies the high-degree coefficients.
        x_shift = galois.Poly([1] + [0] * n_parity, field=GF)  # x^{2t}

        for i in range(n_words):
            msg_poly = galois.Poly(msg[i], field=GF)
            remainder = (msg_poly * x_shift) % self.g  # parity polynomial, degree < 2t

            # Build parity array (right-aligned in n_parity slots)
            parity = GF(np.zeros(n_parity, dtype=int))
            if remainder.degree >= 0:
                r = GF(np.array([int(c) for c in remainder.coeffs]))
                parity[n_parity - len(r):] = r

            code[i, :self.l] = msg[i]      # message in high-degree positions
            code[i, self.l:] = parity       # parity in low-degree positions

        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'
        return code

    def decode(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        alpha = GF.primitive_element
        n_words = np.shape(code)[0]
        n_parity = self.n - self.k  # 2t
        cw_len = self.l + n_parity

        decoded = GF(np.zeros((n_words, self.l), dtype=int))
        nERR = np.zeros(n_words, dtype=int)

        for i in range(n_words):
            word = code[i]

            # --- Step 1: compute syndromes S_j = r(alpha^{m0+j}) for j=0..2t-1 ---
            r_poly = galois.Poly(word, field=GF)
            S = GF([int(r_poly(alpha ** (self.m0 + j))) for j in range(n_parity)])

            if np.all(S == 0):
                decoded[i] = word[:self.l]
                nERR[i] = 0
                continue

            try:
                # --- Step 2: Berlekamp-Massey algorithm (Non-binary codes slides: 206-208) ---
                # Returns Lambda(z): error locator polynomial, Lambda(z) = 1 + Lambda_1*z + ... + Lambda_nu*z^nu
                # Roots of Lambda(z) are alpha^{-l_i} for each error position l_i
                Lambda = RSCode._berlekamp_massey(S, GF, self.t)
                nu = Lambda.degree  # number of errors

                # --- Step 3: Chien search – find array positions of errors ---
                # Array position p (0-indexed, left=highest degree) corresponds to exponent l_i = cw_len-1-p.
                # Lambda has root at alpha^{-l_i} for each error position p.
                error_pos = []
                for p in range(cw_len):
                    exp = cw_len - 1 - p
                    if Lambda(alpha ** (-exp)) == GF(0):
                        error_pos.append(p)

                if len(error_pos) != nu:
                    raise ValueError(f"Chien search found {len(error_pos)} roots, expected {nu}")

                # --- Step 4: Forney algorithm (Non-binary codes slide 223) – compute error values ---
                # S(z) = S_0 + S_1*z + ... + S_{2t-1}*z^{2t-1} (galois: highest degree first = reversed)
                S_poly = galois.Poly([int(c) for c in reversed(S)], field=GF)

                # Error evaluator: Omega(z) = S(z) * Lambda(z) mod z^{2t}
                x_2t = galois.Poly([1] + [0] * n_parity, field=GF)
                Omega = (S_poly * Lambda) % x_2t

                # Formal derivative Lambda'(z) (in GF(2^m) char 2: only odd-degree terms survive)
                Lambda_deriv = RSCode._formal_derivative(Lambda, GF)

                # Apply corrections: e_{l_i} = -alpha^{l_i*(1-m0)} * Omega(alpha^{-l_i}) / Lambda'(alpha^{-l_i})
                # In GF(2^m) with characteristic 2: -1 = 1
                corrected = GF(np.array([int(w) for w in word]))
                for p in error_pos:
                    exp = cw_len - 1 - p       # l_i: exponent of error locator
                    alpha_l  = alpha **  exp    # alpha^{l_i}
                    alpha_nl = alpha ** (-exp)  # alpha^{-l_i}

                    denom = Lambda_deriv(alpha_nl)
                    if denom == GF(0):
                        raise ValueError("Zero denominator in Forney formula")

                    e_p = (alpha_l ** (1 - self.m0)) * Omega(alpha_nl) / denom
                    corrected[p] = corrected[p] + e_p  # addition = subtraction in GF(2^m)

                # Verify: all syndromes of corrected word must be zero
                c_poly = galois.Poly(corrected, field=GF)
                S_check = GF([int(c_poly(alpha ** (self.m0 + j))) for j in range(n_parity)])
                if not np.all(S_check == 0):
                    raise ValueError("Syndrome check failed after correction")

                decoded[i] = corrected[:self.l]
                nERR[i] = len(error_pos)

            except Exception:
                # Decoding failed: return raw message portion and signal failure
                decoded[i] = word[:self.l]
                nERR[i] = -1

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)


    @staticmethod
    def _berlekamp_massey(S, GF, t):
        """Berlekamp-Massey algorithm for finding the error locator polynomial.
        Follows the iterative BMA as described in the lecture notes (pages 206-208).

        Args:
            S: GF array of syndromes [S_0, S_1, ..., S_{2t-1}]
            GF: the Galois field class
            t: error correction capability

        Returns:
            Lambda: galois.Poly, the error locator polynomial Lambda(z) = 1 + Lambda_1*z + ... + Lambda_nu*z^nu
                    Roots of Lambda(z) are alpha^{-l_i} for each error position l_i.
        """
        two_t = len(S)

        # Initialization (PDF page 206):
        #   Lambda^(0)(z) = 1,  B(z) = z,  L_0 = 0
        Lambda = galois.Poly([1], field=GF)     # Lambda^(0)(z) = 1
        B      = galois.Poly([1, 0], field=GF)  # B(z) = z
        L = 0                                    # LFSR length
        z_poly = galois.Poly([1, 0], field=GF)  # z, used for B(z) <- z*B(z)

        for i in range(1, two_t + 1):
            # Step 3: compute discrepancy Delta^(i) = sum_{j=0}^{L} Lambda_j^(i-1) * S_{i-1-j}
            # (Lambda_0 = 1 always; Lambda_j is coefficient of z^j in Lambda)
            delta = GF(0)
            for j in range(L + 1):
                # coeff of z^j in Lambda: stored at index Lambda.degree - j (highest-first)
                lam_j = GF(int(Lambda.coeffs[Lambda.degree - j])) if j <= Lambda.degree else GF(0)
                delta = delta + lam_j * S[i - 1 - j]

            Lambda_prev = Lambda  # save for potential B update

            if delta != GF(0):
                # Step 4b: Lambda^(i)(z) = Lambda^(i-1)(z) - Delta^(i) * B(z)
                Lambda = Lambda - GF(int(delta)) * B

                if 2 * L < i:
                    # Must increase register length (PDF page 207, left column)
                    L = i - L
                    B = Lambda_prev * (GF(int(delta)) ** -1)

                # Step 6: if degree exceeds t, cannot decode
                if Lambda.degree > t:
                    raise ValueError(f"Too many errors: deg(Lambda)={Lambda.degree} > t={t}")

            # Step 5: B(z) <- z * B(z)
            B = z_poly * B

        # Step 7: return Lambda(z) = Lambda^(2t)(z)
        return Lambda

    @staticmethod
    def _formal_derivative(poly, GF):
        """Formal derivative of a polynomial over GF(2^m).

        In characteristic 2, d/dx(c * x^k) = 0 for even k and c * x^{k-1} for odd k.
        """
        degree = poly.degree
        if degree <= 0:
            return galois.Poly([GF(0)], field=GF)

        # poly.coeffs is [c_degree, c_{degree-1}, ..., c_0] (highest first)
        # coefficient of x^k is poly.coeffs[degree - k]
        result = {}  # maps power-in-derivative -> GF coefficient
        for k in range(1, degree + 1):
            if k % 2 == 1:               # only odd k survive in char 2
                c_k = poly.coeffs[degree - k]
                result[k - 1] = c_k      # x^k term becomes x^{k-1} in derivative

        if not result:
            return galois.Poly([GF(0)], field=GF)

        max_pow = max(result.keys())
        # Build coefficient list highest-first (galois convention)
        coeffs = [result.get(p, GF(0)) for p in range(max_pow, -1, -1)]
        return galois.Poly([int(c) for c in coeffs], field=GF)

    @staticmethod
    def makeGenerator(m, t, m0):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(2^m)
        # Input:
        #  -m: order of the galois field is 2^m
        #  -t: error correction capability of the Reed-Solomon code
        #  -m0: determines the first root of the generator polynomial
        # Output:
        #  -generator: generator polynomial represented by a galois.Poly variable

        GF    = galois.GF(2**m)
        alpha = GF.primitive_element  # primitive element alpha of GF(2^m)

        # g(x) = prod_{i=0}^{2t-1} (x - alpha^{m0+i})
        # In GF(2^m) characteristic 2: (x - a) = (x + a), so the factor is [1, int(a)]
        g = galois.Poly([1], field=GF)
        for i in range(2 * t):
            root   = alpha ** (m0 + i)
            factor = galois.Poly([1, int(root)], field=GF)  # x + root = x - root in char 2
            g      = g * factor

        generator = g
        assert type(generator) == type(galois.Poly([0],field=galois.GF(2**m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def test():
        # function that illustrates how the other code of this class can be tested
        m0 = 1 # Also test with other values of m0!
        m=8
        t=5
        l=10
        rs = RSCode(m,t,l,m0) # Construct the RSCode object
        p=2
        prim_poly=galois.primitive_poly(p,m)
        galois_field=galois.GF(p**m, irreducible_poly=prim_poly)


        msg = galois_field(np.random.randint(0,2**8-1,(5,10))) # Generate a random message of 5 information words

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**8-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**8-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass


if __name__ == "__main__":
    RSCode.test()