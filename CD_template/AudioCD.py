import copy
import math
import struct
import wave
import numpy as np
from playsound import playsound
from reedsolo import RSCodec

class AudioCD:

    def __init__(self, Fs,configuration,max_interpolation):
        self.Fs=Fs # Sample rate of the audio
        self.max_interpolation=max_interpolation # The maximum number of interpolated audio samples
        self.number_of_errors=0
        self.number_of_errors_corrected=0
        self.number_of_uncorrectedC1=0
        self.cd_bits=[] #Bits written to disk (before EFM)
        self.cd_bits_original=[]
        self.scaled_quantized_padded_original=[] #Reference to compare the output of readCD to


        # initialise encoders/decoders
        if configuration==1 or configuration==2:
            self.rsc2 = RSCodec( nsym=4, nsize=255, fcr=0, prim=0x11d, generator=2, c_exp=8, single_gen=True)
            self.rsc1 = RSCodec( nsym=4, nsize=255, fcr=0, prim=0x11d, generator=2, c_exp=8, single_gen=True)
        elif configuration==3:
            self.rsc3 = RSCodec( nsym=8, nsize=255, fcr=0, prim=0x11d, generator=2, c_exp=8, single_gen=True)

        self.configuration = configuration # 0: no CIRC; 1: CIRC as described in standard; 2: Concatenated RS, no interleaving; 3: Single 32,24 RS

    def save_and_play_music(self, left_channel,right_channel, wav_file, bool_play=1):
        # this function transforms the left and right channel back to playable wav file (see the test() function how this function can be used)
        # Input:
        #  -left_channel/right_channel: a 1D numpy ndarray that contains the data of an audio file
        #  -wav_file: filename of the audiofile that will be created
        #  -bool_Play: bool that determines if the audio file needs to be played
        assert len(np.shape(left_channel))==1 and len(np.shape(right_channel))==1 and type(left_channel) is np.ndarray and type(right_channel) is np.ndarray, 'the left and right channel must be 1D numpy arrays'

        data=np.zeros(len(left_channel)+len(right_channel))
        data[0::2]=left_channel.flatten()
        data[1::2]=right_channel.flatten()
        data=np.round(data*(2**15))

        data=data.astype(int)
        wave_object = wave.open(wav_file, 'wb')
        wave_object.setnchannels(2)
        wave_object.setsampwidth(2)
        wave_object.setframerate(self.Fs)
        n_length = len(data)
        for i in range(n_length):
            value = data[i]

            value_packed = struct.pack('<h', max(min(32767,value),-32768))
            wave_object.writeframesraw(value_packed)
        wave_object.close()
        if bool_play:
            try:
                playsound(wav_file)
            except:
                pass

        pass

    def writeCd(self,audiofile):
        # Write an audiofile to the CD
        # Input:
        #  -audiofile: an Nsamples x 2 numpy array containing the left and right audio track as samples
        assert np.shape(audiofile)[1]==2 and type(audiofile) is np.ndarray, 'audiofile must be a 2D numpy array with 2 columns'


        xscaled = audiofile / np.max(np.abs(audiofile)) # normalize to -1:1

        x=self.uencode(xscaled) #convert to 16 bit signed values

        xlr16 = np.reshape(np.transpose(x),(-1,1),order='F') # serialize left and right audio channel

        xlr8=self.typecast_8(xlr16)#split into 8 bit words

        xlr8_padded = np.hstack((xlr8 , np.zeros((24-(np.fmod(xlr8.size-1,24)+1)))) )# pad with zeros to fill an integer number of frames

        n_frames = xlr8_padded.size//24 # every frame contains 24 8 bit words

        ylr16=self.typecast_16(xlr8_padded)

        y = np.transpose(np.reshape(ylr16,(2,-1),order='F'))

        self.scaled_quantized_padded_original = self.udecode(y) # Reference to compare the output of readCD to

        if self.configuration==0: # no CIRC
            encoded=xlr8_padded.astype('B')
        elif self.configuration==1: # CIRC as described in standard
            (delay_interleaved,n_frames) = self.CIRC_enc_delay_interleave(xlr8_padded,n_frames)
            (C2_encoded,n_frames) = self.CIRC_enc_C2(delay_interleaved,n_frames)
            (delay_unequal,n_frames) = self.CIRC_enc_delay_unequal(C2_encoded,n_frames)
            (C1_encoded,n_frames) = self.CIRC_enc_C1(delay_unequal,n_frames)
            (delay_inv,n_frames) = self.CIRC_enc_delay_inv(C1_encoded,n_frames)
            encoded=delay_inv

        elif self.configuration==2: #Concatenated RS, no interleaving
            (C2_encoded,n_frames) = self.CIRC_enc_C2(xlr8_padded,n_frames)
            (C1_encoded,n_frames) = self.CIRC_enc_C1(C2_encoded,n_frames)
            encoded=C1_encoded


        elif self.configuration==3: #Single 32,24 RS
            (encoded,n_frames) = self.C3_enc_8_parity(xlr8_padded,n_frames)

        else:
            print('Invalid configuration selected')
            exit(-1)

        xlrbserial=np.unpackbits(encoded, bitorder='little')

        self.cd_bits = copy.deepcopy(xlrbserial)
        self.cd_bits_original= copy.deepcopy(xlrbserial)

        return

    def bitErrorsCd(self,p):
        # Add uniform bit errors to cd
        # Input:
        #  -p: the bit error probability, i.e., a self.cd_bits bit is flipped with probability p
        noise = np.random.rand((self.cd_bits).shape)<p
        self.cd_bits = np.bitwise_xor(self.cd_bits,noise.astype(int))
        return

    def scratchCd(self,length_scratch,location_scratch):
        # Add a scratch to the cd
        # Input:
        #  -length_scratch: the length of the scratch (in number of bits)
        #  -location_scratch: the location of the scratch (in bits offset from start of self.cd_bits)
        self.cd_bits[location_scratch-1:min(location_scratch-1+length_scratch,(self.cd_bits).size)] = 0
        self.number_of_errors=np.sum(self.cd_bits!=self.cd_bits_original)
        return

    def readCd(self):
        # Read an audiofile from the CD
        # Output:
        #  -audio_out: an Nsamples x 2 numpy array containing the left and right audio track as samples
        #  -interpolation_flags: an Nsamples x 2 numpy array containing a 0 where no erasure was flagged, a 1 where an erasure was interpolated and a -1 where interpolation failed


        ylr8=np.packbits(self.cd_bits, bitorder='little')

        if self.configuration== 0: # no CIRC
            ylr16 = self.typecast_16(ylr8)
            y = np.transpose(np.reshape(ylr16,(2,-1),order='F'))
            audio_out = self.udecode(y)
            interpolation_flags = np.zeros(np.shape(audio_out))
        elif self.configuration== 1: # CIRC as described in standard
            n_frames = ylr8.size/32
            assert(n_frames*32 == ylr8.size)

            (delay_inv,n_frames) = self.CIRC_dec_delay_inv(ylr8,n_frames)
            (C1_decoded,erasure_flags,n_frames) = self.CIRC_dec_C1(delay_inv,n_frames)
            (delay_unequal,erasure_flags,n_frames) = self.CIRC_dec_delay_unequal(C1_decoded,erasure_flags,n_frames)
            (C2_decoded,erasure_flags,n_frames) = self.CIRC_dec_C2(delay_unequal,erasure_flags,n_frames)
            (deinterleave_delay,erasure_flags,n_frames) = self.CIRC_dec_deinterleave_delay(C2_decoded,erasure_flags,n_frames)

            ylr16 = self.typecast_16(deinterleave_delay)
            y = np.transpose(np.reshape(ylr16,(2,-1),order='F'))

            erasure_flags = np.reshape(erasure_flags,(2,-1),order='F')
            erasure_flags = np.transpose(np.logical_or(erasure_flags[0,:],erasure_flags[1,:]))
            erasure_flags = np.transpose(np.reshape(erasure_flags,(2,-1),order='F'))

            # Linear Interpolation
            interpolation_failed = np.zeros(np.shape(erasure_flags),)
            (y[:,0],interpolation_failed[:,0]) = self.interpolator(y[:,0],erasure_flags[:,0]) # Left
            (y[:,1],interpolation_failed[:,1]) = self.interpolator(y[:,1],erasure_flags[:,1]) # Right

            audio_out = self.udecode(y)
            interpolation_flags = np.zeros(np.shape(audio_out))
            interpolation_flags[erasure_flags] = 1
            interpolation_flags[interpolation_failed.astype(bool)] = -1


        elif self.configuration== 2: # Concatenated RS, no interleaving
            n_frames = ylr8.size/32
            assert(n_frames*32 == ylr8.size)

            (C1_decoded,erasure_flags,n_frames) = self.CIRC_dec_C1(ylr8,n_frames)
            erasure_flags_t = erasure_flags
            (C2_decoded,erasure_flags,n_frames) = self.CIRC_dec_C2(C1_decoded,erasure_flags,n_frames)

            if(erasure_flags.size  != C2_decoded.size):
                print('Something wrong!')


            ylr16 = self.typecast_16(C2_decoded)
            y = np.transpose(np.reshape(ylr16,(2,-1),order='F'))


            erasure_flags = np.reshape(erasure_flags,(2,-1),order='F')
            erasure_flags = np.transpose(np.logical_or(erasure_flags[0,:],erasure_flags[1,:]))
            erasure_flags = np.transpose(np.reshape(erasure_flags,(2,-1),order='F'))

            #  Linear Interpolation
            interpolation_failed = np.zeros(np.shape(erasure_flags))
            (y[:,0],interpolation_failed[:,0]) = self.interpolator(y[:,0],erasure_flags[:,0]) # Left
            (y[:,1],interpolation_failed[:,1]) = self.interpolator(y[:,1],erasure_flags[:,1]) # Right

            audio_out = self.udecode(y)
            interpolation_flags = np.zeros(np.shape(audio_out))
            interpolation_flags[erasure_flags] = 1
            interpolation_flags[interpolation_failed.astype(bool)] = -1

        elif self.configuration== 3:# Single 32,24 RS
            n_frames = ylr8.size/32
            assert(n_frames*32 == ylr8.size)

            (decoded,erasure_flags,n_frames) = self.C3_dec_8_parity(ylr8,n_frames)
            ylr16 = self.typecast_16(decoded)
            y = np.transpose(np.reshape(ylr16,(2,-1),order='F'))

            erasure_flags = np.reshape(erasure_flags,(2,-1),order='F')
            erasure_flags = np.transpose(np.logical_or(erasure_flags[0,:],erasure_flags[1,:]))
            erasure_flags = np.transpose(np.reshape(erasure_flags,(2,-1),order='F'))

            # Linear Interpolation
            interpolation_failed = np.zeros(np.shape(erasure_flags))
            ([y[:,0],interpolation_failed[:,0]]) = self.interpolator(y[:,0],erasure_flags[:,0]) # Left
            ([y[:,1],interpolation_failed[:,1]]) = self.interpolator(y[:,1],erasure_flags[:,1]) # Right

            audio_out = self.udecode(y)
            interpolation_flags = np.zeros(np.shape(audio_out))
            interpolation_flags[erasure_flags] = 1
            interpolation_flags[interpolation_failed.astype(bool)] = -1

        else:
            print('Invalid configuration selected')
            exit(1)

        assert np.shape(audio_out)[1]==2 and type(audio_out) is np.ndarray, 'audio_out must be a 2D numpy array with 2 columns'
        assert np.shape(interpolation_flags)[1]==2 and type(interpolation_flags) is np.ndarray, 'interpolation_flags must be a 2D numpy array with 2 columns'
        return (audio_out,interpolation_flags)

    def CIRC_enc_delay_interleave(self,input,n_frames): 
        # CIRC Encoder: Delay of 2 frames + interleaving sequence
        # Input:
        #  -input: the input to this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
        
        # delay of 2 frames, so the output is 2 frames longer than the input
        n_frames_output = n_frames + 2 

        output_temp = np.zeros(n_frames_output * 24)

        
        # delay all even numbered symbols
        for n in range(n_frames):
            for i in range(24):                                         # 24 symbols of 8 bits
                if (i // 4) % 2 == 0:                                   # even samples have a delay of 2 frames 
                    output_temp[(n + 2) * 24 + i] = input[n * 24 + i]   # 1 frame = 24 symbols, 2 frames = 2*24 = 48 symbols
                else:
                    output_temp[n * 24 + i] = input[n * 24 + i]         #uneven samples are not delayed

        # interleave
        output = np.zeros(n_frames_output * 24)
        for n in range(n_frames_output):
            for i in range(12): 
                # even and odd datawords are separated
                output[n * 24 + i] = output_temp[n * 24 + i * 2]
                output[n * 24 + i + 12] = output_temp[n * 24 + i * 2 + 1]
        #print(output[0:20])

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def CIRC_enc_C2(self,input,n_frames): 
        # CIRC Encoder: Generation of 4 parity symbols (C2)
        # Input:
        #  -input: the input to this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the output expressed in frames

        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
                
        input=input.astype('B')                                 # input: 24 symbols 
        output = np.zeros(int(n_frames*28),dtype='B')           # (28,24) RS code, 28 = 24 symbols + 4 parity symbols

        for n in range(n_frames):
            input_frame = input[n*24:(n+1)*24]                  # take one input frame
            encoded_frame = self.rsc2.encode(input_frame)       # encode this input frame
            output[n*28:n*28+12] = encoded_frame[0:12]          # first 12 symbols of encoded frame
            output[n*28+12:n*28+16] = encoded_frame[24:28]      # middel 4 symbols are the 4 Q parity symbols
            output[n*28+16:n*28+28] = encoded_frame[12:24]      # last 12 symbols of encoded frame

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)                                # output has same number of frames as input

    def CIRC_enc_delay_unequal(self,input,n_frames):
        # CIRC Encoder: Delay lines of unequal length
        # Input:
        #  -input: the input to this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        #insert your code here
        input = input.astype('B')
        D = 4                                                   # inter-symbol delay unit in frames (from the standard)

        n_frames_output = n_frames + D*27                       # maximum delay of 27*D (=27*4=108)
        output = np.zeros(n_frames_output * 28, dtype = 'B')

        for n in range(n_frames):
            for i in range(28):
                delay = i*D
                output[(n+delay)*28 + i] = input[n*28 + i]      # delay of 0, 4, 8, ..., 108 frames

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def CIRC_enc_C1(self,input,n_frames):
        # CIRC Encoder: Generation of 4 parity symbols (C1)
        # Input:
        #  -input: the input to this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the output expressed in frames
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        #insert your code here
        input=input.astype('B')                                 # input: 28 symbols 
        output = np.zeros(int(n_frames*32),dtype='B')           # (32,28) RS code, 32 = 28 symbols + 4 P parity symbols

        for n in range(n_frames):
            input_frame = input[n*28:(n+1)*28]                  # take one input frame
            encoded_frame = self.rsc2.encode(input_frame)       # encode this input frame
            output[n*32:(n+1)*32] = encoded_frame[0:32]         # order here is correct, four P parity symbols at the end 

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def CIRC_enc_delay_inv(self,input,n_frames):
        # CIRC Encoder: Delay of 1 frame + inversions
        # Input:
        #  -input: the input to this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC encoder (1D numpy array)
        #  -n_frames: the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        #insert your code here
        input = input.astype('B')
        n_frames_output = n_frames + 1                          # delay 1
        output = np.zeros(n_frames_output * 32, dtype = 'B')

        for n in range(n_frames):
            for i in range(32):
                if i % 2 == 0:                                  # symbols with even index are delayed by 1
                    output[(n+1)*32 + i] = input[n*32 + i]      # even byte of C1 frame n  →  disc frame n+1
                else:                                           # symbols with odd index are not delayed
                    output[n*32 + i] = input[n*32 + i]          # odd  byte of C1 frame n  →  disc frame n
        
        #inverting parity symbols
        for n in range(n_frames_output):
            for i in [12,13,14,15,28,29,30,31]:                 # positions of parity symbols (Q parity at 12-15, P parity at 28-31)
                output[n*32 + i] = output[n*32 + i] ^ 0xFF      # inversion of parity symbols (XOR with 0xFF)

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def CIRC_dec_delay_inv(self,input,n_frames):
        # CIRC Decoder: Delay of 1 frame + inversions
        # Input:
        #  -input: the input to this block of the CIRC decoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC decoder (1D numpy array)
        #  -n_frames:  the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        input = input.copy().astype('B')

        # inverting parity symbols
        for n in range(int(n_frames)):
            for i in [12, 13, 14, 15, 28, 29, 30, 31]:          # positions of parity symbols (Q parity at 12-15, P parity at 28-31)
                input[n*32 + i] ^= 0xFF                         # inversion of parity symbols (XOR with 0xFF)

        # delay of 1 frame -> undo 1-frame delay on even-indexed symbols
        # encoder delayed all even-indexed bytes of each C1 codeword by 1 disc frame
        # decoder undoes this delay
        n_frames_output = int(n_frames) - 1                     # undo 1-frame delay -> reduces frame count by 1
        output = np.zeros(n_frames_output * 32, dtype='B')

        for n in range(n_frames_output):
            for i in range(32):                                 # 32 symbols per frame
                if i % 2 == 0:                                  # even-indexed symbols came from the next disc frame
                    output[n*32 + i] = input[(n+1)*32 + i]      # even byte from disc frame n+1 (advance by 1)
                else:                                           # odd-indexed symbols came from the current disc frame
                    output[n*32 + i] = input[n*32 + i]          # odd  byte from disc frame n (no shift)

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def CIRC_dec_C1(self,input,n_frames):
        # CIRC Decoder: C1 decoder
        # Input:
        #  -input: the input to this block of the CIRC decoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_out: the erasure flags at the output of this block, follow the decoding algorithm from the assignment (1D numpy array)
        #  -n_frames: the length of the output expressed in frames
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        input = input.astype('B')
        output = np.zeros(int(n_frames) * 28, dtype='B')            # C1 decoder output consists of 28 symbols
        erasure_flags_out = np.zeros(int(n_frames) * 28)

        # C1 is a (32,28) RS code over GF(2^8) with 4 P-parity symbols:
        # minimum distance of 5 and an error-correction capability of t=2.

        # CIRC decoding policy for C1:
        #   0 errors detected : pass through unchanged, no flag
        #   1 error  detected : correct and pass through, no flag
        #   2+ errors detected: do NOT correct (unreliable), flag entire frame for C2
        #   decoding failure  : flag entire frame for C2

        # flagging with 2 errors (rather than correcting) is intentional: correcting
        # 2 errors uses all 4 parities, leaving no redundancy to verify the correction.
        # flagging instead lets C2 handle the word as an erasure, which is more reliable.

        for n in range(int(n_frames)):

            frame = bytes(input[n*32:(n+1)*32])

            try:
                # output of RSCodec.decode: decoded message, decoded message + error correction code, list of positions of the erata
                decoded, _, err = self.rsc1.decode(frame, erase_pos=None)
                ERR = len(err)          # number of corrected symbol positions
                output_dec = list(decoded)

            except Exception:
                # More errors than C1 can handle: pass through the raw (corrupted) data
                ERR = -1
                output_dec = list(input[n*32 : n*32 + 28])

            output[n*28:(n+1)*28] = output_dec

            if ERR == -1 or ERR >= 2:
                # Flag all 28 symbols of this frame so C2 treats them as erasures
                erasure_flags_out[n*28:(n+1)*28] = 1

        # no delays, n_frames at output remains the same

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(erasure_flags_out))==1 and type(erasure_flags_out) is np.ndarray, 'erasure_flags_out must be a 1D numpy array'
        return (output,erasure_flags_out,n_frames)

    def CIRC_dec_delay_unequal(self,input,erasure_flags_in,n_frames):
        # CIRC Decoder: Delay lines of unequal length
        # Input:
        #  -input: the input to this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_in: the erasure flags at the input of this block of the CIRC decoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_out: the erasure flags at the output of this block, follow the decoding algorithm from the assignment (1D numpy array)
        #  -n_frames:  the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
        assert len(np.shape(erasure_flags_in))==1 and type(erasure_flags_in) is np.ndarray, 'erasure_flags_in must be a 1D numpy array'

        input = input.astype('B')
        D = 4   # inter-symbol delay unit in frames (from the standard)

        # unequal delay to the 28 symbols of each C2 codeword was applied by the encoder:
        # symbol i was delayed by i*D frames before C1 encoding (0 to 27*D=108)
        # inverse:
        # to reassemble C2 codeword m, symbol i must be fetched from C1 decoded frame m+i*D

        n_frames_output = int(n_frames) - D * 27    # maximum delay was 27*D = 108 frames    
        output = np.zeros(n_frames_output * 28, dtype='B')
        erasure_flags_out = np.zeros(n_frames_output * 28)

        for m in range(n_frames_output):
            for i in range(28):
                src = m + i * D     # C1 frame that holds symbol i of C2 codeword m
                output[m*28 + i]          = input[src*28 + i]
                erasure_flags_out[m*28+i] = erasure_flags_in[src*28 + i]

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(erasure_flags_out))==1 and type(erasure_flags_out) is np.ndarray, 'erasure_flags_out must be a 1D numpy array'
        return (output,erasure_flags_out,n_frames)

    def CIRC_dec_C2(self,input,erasure_flags_in,n_frames):
        # CIRC Decoder: C2 decoder
        # Input:
        #  -input: the input to this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_in: the erasure flags at the input of this block of the CIRC decoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_out: the erasure flags at the output of this block, follow the decoding algorithm from the assignment (1D numpy array)
        #  -n_frames: the length of the output expressed in frames
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
        assert len(np.shape(erasure_flags_in))==1 and type(erasure_flags_in) is np.ndarray, 'erasure_flags_in must be a 1D numpy array'

        input = input.astype('B')

        # C2 is a (28,24) RS code over GF(2^8) with 4 Q-parity symbols
        # It can correct up to 4 erasures (symbols whose positions are known from C1 flags)
        # Decoding fails if too many erasures or additional symbol errors
        # Then entire 24-symbol output word is flagged
        
        output = np.zeros(int(n_frames) * 24, dtype='B')
        erasure_flags_out = np.zeros(int(n_frames) * 24)

        for n in range(int(n_frames)):
            frame = input[n*28:(n+1)*28]
            flags = erasure_flags_in[n*28:(n+1)*28]

            # The C2 encoder inserted the 4 Q parity symbols in the centre of the frame (positions 12-15)
            # CIRC layout : [data 0-11 | Q parity 0-3 | data 12-23]  (positions 0-27)
            # reedsolo expects : [data 0-11 | data 12-23 | Q parity 0-3]
            # Reorder the frame and remap erasure positions accordingly
            reordered = np.zeros(28, dtype='B')
            reordered[0:12]  = frame[0:12]    # data symbols 0-11  stay in place
            reordered[12:24] = frame[16:28]   # data symbols 12-23 moved forward
            reordered[24:28] = frame[12:16]   # Q parity moved to the end

            # Map erasure positions from CIRC layout to reedsolo layout:
            # CIRC pos  0-11 (data)   → reedsolo pos  0-11  (unchanged)
            # CIRC pos 12-15 (Q par.) → reedsolo pos 24-27  (+12)
            # CIRC pos 16-27 (data)   → reedsolo pos 12-23  (-4)
            erase_pos = []
            for j in range(28):
                if flags[j]:
                    if j < 12:
                        erase_pos.append(j)       # data 0-11  → same position
                    elif j < 16:
                        erase_pos.append(j + 12)  # Q parity   → positions 24-27
                    else:
                        erase_pos.append(j - 4)   # data 12-23 → positions 12-23

            try:
                decoded, _, err = self.rsc2.decode(bytes(reordered),
                                                   erase_pos=erase_pos if erase_pos else None)
                ERR = len(err)              # number of corrected positions (errors + erasures)
                output_dec = list(decoded)  # 24 decoded data bytes

            except Exception:
                # C2 could not correct: too many erasures or additional symbol errors.
                # Pass through the raw (corrupted) data bytes and flag for interpolation.
                ERR = -1
                output_dec = list(frame[0:12]) + list(frame[16:28])

            output[n*24:(n+1)*24] = output_dec
            if ERR == -1:
                # Flag all 24 output symbols so the interpolator can attempt concealment
                erasure_flags_out[n*24:(n+1)*24] = 1

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(erasure_flags_out))==1 and type(erasure_flags_out) is np.ndarray, 'erasure_flags_out must be a 1D numpy array'
        return (output,erasure_flags_out,n_frames)

    def CIRC_dec_deinterleave_delay(self,input,erasure_flags_in,n_frames):
        # CIRC Decoder: De-interleaving sequence + delay of 2 frames
        # Input:
        #  -input: the input to this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_in: the erasure flags at the input of this block of the CIRC decoder (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block of the CIRC decoder (1D numpy array)
        #  -erasure_flags_out: the erasure flags at the output of this block, follow the decoding algorithm from the assignment (1D numpy array)
        #  -n_frames:  the length of the output expressed in frames (changed from input because of delay!)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
        assert len(np.shape(erasure_flags_in))==1 and type(erasure_flags_in) is np.ndarray, 'erasure_flags_in must be a 1D numpy array'

        input = input.astype('B')

        # De-interleave:
        # Encoder interleaved the 24 bytes of each frame by separating even- and odd-indexed bytes into two groups, placing them in consecutive half-frames
        # Physically separates the even and odd audio samples on disc so that burst error destroying several adjacent bytes only affects one sample from each stereo pair, making concealment by interpolation more effective. 
        # Encoder interleave:
        #   even byte positions (0,2,4,...,22) of a frame → output positions 0-11
        #   odd  byte positions (1,3,5,...,23) of a frame → output positions 12-2
        # Inverse (de-interleave):
        #   input positions  0-11 → even byte positions (0,2,...,22) of temp frame
        #   input positions 12-23 → odd  byte positions (1,3,...,23) of temp frame
        output_temp = np.zeros(int(n_frames) * 24, dtype='B')
        flags_temp  = np.zeros(int(n_frames) * 24)
        for n in range(int(n_frames)):
            for i in range(12):
                output_temp[n*24 + i*2]     = input[n*24 + i]               # even positions
                output_temp[n*24 + i*2 + 1] = input[n*24 + i + 12]          # odd positions
                flags_temp[n*24 + i*2]      = erasure_flags_in[n*24 + i]
                flags_temp[n*24 + i*2 + 1]  = erasure_flags_in[n*24 + i + 12]

        # Undo 2-frame delay on even-word bytes
        # Encoder delayed even-numbered audio words (bytes where (i//4)%2 == 0) by 2 frames before interleaving.  This separates consecutive audio samples on disc so that an uncorrectable burst destroying several frames leaves alternating valid and invalid samples, enabling linear interpolation.

        # Encoder delay mapping:  original frame n, byte i (even-word) → temp frame n+2
        #                         original frame n, byte i (odd-word)  → temp frame n
        # Inverse: original frame m, byte i =
        #   temp frame m+2, byte i   if (i//4)%2 == 0  (even-word bytes were delayed)
        #   temp frame m,   byte i   otherwise

        n_frames_output = int(n_frames) - 2 # output has 2 fewer frames because undoing the 2-frame delay
        output = np.zeros(n_frames_output * 24, dtype='B')
        erasure_flags_out = np.zeros(n_frames_output * 24)

        for m in range(n_frames_output):
            for i in range(24):
                if (i // 4) % 2 == 0:   # even-word byte: was delayed, fetch from 2 frames ahead
                    output[m*24 + i]          = output_temp[(m+2)*24 + i]
                    erasure_flags_out[m*24+i] = flags_temp[(m+2)*24 + i]
                else:                    # odd-word byte: not delayed, fetch from current frame
                    output[m*24 + i]          = output_temp[m*24 + i]
                    erasure_flags_out[m*24+i] = flags_temp[m*24 + i]

        n_frames = n_frames_output

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(erasure_flags_out))==1 and type(erasure_flags_out) is np.ndarray, 'erasure_flags_out must be a 1D numpy array'
        return (output,erasure_flags_out,n_frames)

    def C3_enc_8_parity(self,input,n_frames):
        # Configuration 3: Generation of 8 parity symbols
        # Input:
        #  -input: the input to this block (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block (1D numpy array)
        #  -n_frames: the length of the output expressed in frames
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        input=input.astype('B')
        output = np.zeros(int(n_frames*32),dtype='B')

        for i in range(int(n_frames)):
            encoded= self.rsc3.encode(input[(i)*24:(i+1)*24])
            encoded=list(encoded)
            output[(i)*32:(i+1)*32] = encoded

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

    def C3_dec_8_parity(self,input,n_frames):
        # Configuration 3: Decoder
        # Input:
        #  -input: the input of this block (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        # Output:
        #  -output: the output of this block (1D numpy array)
        #  -erasure_flags_out: the erasure flags at the output of this block (1D numpy array)
        #  -n_frames: the length of the input expressed in frames
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'

        input=input.astype('B')
        output = np.zeros(int(n_frames*24),dtype='B')
        erasure_flags_out = np.zeros(int(n_frames*24))
        for i in range(int(n_frames)):
            try:
                (decoded,_,err)=self.rsc3.decode(input[(i)*32:(i+1)*32],erase_pos=None)
                ERR=len(err)
                output_dec=list(decoded)
                output_dec=output_dec[-24:]
            except Exception as e:
                ERR=-1
                output_dec=input[(i)*32:(i)*32+24]

            if ERR == -1:
                output[(i)*24:(i+1)*24] = output_dec
                erasure_flags_out[(i)*24:(i+1)*24] = 1
            else:
                output[(i)*24:(i+1)*24] = output_dec

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(erasure_flags_out))==1 and type(erasure_flags_out) is np.ndarray, 'erasure_flags_out must be a 1D numpy array'
        return (output,erasure_flags_out,n_frames)

    def interpolator(self,input,erasure_flags_in):
        # Interpolation: Linear interpolation
        # Input:
        #  -input: the input to this block (1D numpy array)
        #  -erasure_flags_in: the erasure flags at the input of this block (1D numpy array)
        # Output:
        #  -output: linear interpolation of the input where there are no more than self.max_interpolation consecutive erasures (1D numpy array)
        #  -interpolation_failed: equal to one at the samples where interpolation failed (1D numpy array)
        assert len(np.shape(input))==1 and type(input) is np.ndarray, 'input must be a 1D numpy array'
        assert len(np.shape(erasure_flags_in))==1 and type(erasure_flags_in) is np.ndarray, 'erasure_flags_in must be a 1D numpy array'

        erasure_flags_in=erasure_flags_in.astype(int)
        if erasure_flags_in[0] != 0:
            erasure_flags_in[0] = 0
            input[0] = 2^15

        if erasure_flags_in[-1] != 0:
            erasure_flags_in[-1] = 0
            input[-1] = 2^15

        output = copy.deepcopy(input)
        interpolation_failed = copy.deepcopy(erasure_flags_in)


        erasure_burst = np.zeros(erasure_flags_in.size,dtype='B') # Number of consecutive erasures
        ii=np.where(np.diff(erasure_flags_in)==1)[0]+1

        if len(ii)!=0:
            erasure_burst[ii] = np.where(np.diff(erasure_flags_in)==-1)[0] - np.asarray(ii)+1
            temp=np.where((erasure_burst>0) & (erasure_burst<= self.max_interpolation) )[0]
            if len(temp)>0:
                for i in temp:
                    output[i:i+erasure_burst[i]] = np.maximum(np.zeros(erasure_burst[i]),np.minimum((2**16-1)*np.ones(erasure_burst[i]),(np.round(float(output[i-1])+np.arange(0,erasure_burst[i])*(float(output[i+erasure_burst[i]])-float(output[i-1]))/(erasure_burst[i]+1))).astype(int)))
                    interpolation_failed[i:i+erasure_burst[i]] = 0


        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        assert len(np.shape(interpolation_failed))==1 and type(interpolation_failed) is np.ndarray, 'interpolation_failed must be a 1D numpy array'
        return (output, interpolation_failed)

    @staticmethod
    def uencode(xscaled):
        delta=2/(2**16-1)
        x=np.round((1+xscaled)/delta) # convert to 16 bit signed values
        return x

    @staticmethod
    def udecode(y):
        delta=2/(2**16-1)
        x=-1+y*delta
        return x

    @staticmethod
    def typecast_8(xlr16):
        xlr8=np.zeros(len(xlr16)*2)
        temp1=np.mod(xlr16,256)
        temp2=np.floor_divide(xlr16,256)
        xlr8[::2]=temp1.flatten()
        xlr8[1::2]=temp2.flatten()
        return xlr8

    @staticmethod
    def typecast_16(xlr8_padded):
        xlr8_padded = np.asarray(xlr8_padded, dtype=np.uint16)
        ylr16=xlr8_padded[::2] + (2**8)*xlr8_padded[1::2]
        return ylr16

    @staticmethod
    def test():
        # % Test the code of this class
        wave_object = wave.open('Hallelujah.wav','rb')
        number_frames = wave_object.getnframes()
        Fs = wave_object.getframerate()
        nch=wave_object.getnchannels()
        depth = wave_object.getsampwidth()
        wave_object.setpos(0)
        sdata = wave_object.readframes(wave_object.getnframes())
        typ = { 1: np.int8, 2: np.int16, 4: np.int32 }.get(depth)
        if not typ:
            raise ValueError("sample width {} not supported".format(depth))

        data = np.frombuffer(sdata, dtype=typ)
        data=data/(2**15)
        ch_1 = data[0::nch]
        ch_2 = data[1::nch]
        audiofile=np.transpose(np.vstack((ch_1,ch_2)))
        cd = AudioCD(Fs,1,8)
        cd.writeCd(audiofile)
        T_scratch = 600000 # Scratch at a diameter of approx. 66 mm
        #USE scratch_lengths = [0, 3000, 4000, 5000, 10000] to test the performance of the code with different scratch lengths. You can also use scratch_lengths = [0] to test the code without scratches.
        scratch_lengths = [0, 3000, 4000, 5000, 10000]
        #scratch_lengths = [0]
        print(f'\n{"Scratch (bits)":<16} {"Erasures":<12} {"Interpolated":<14} {"Failed":<10} {"Undetected"}')
        print('-' * 65)
        for l_scratch in scratch_lengths:
            import copy
            cd_test = copy.deepcopy(cd)
            for i in range(math.floor(cd_test.cd_bits.size / T_scratch)):
                cd_test.scratchCd(l_scratch, 30000 + i * T_scratch)
            [out, interpolation_flags] = cd_test.readCd()
            n_erasures     = int(np.sum(interpolation_flags != 0))
            n_interpolated = int(np.sum(interpolation_flags ==  1))
            n_failed       = int(np.sum(interpolation_flags == -1))
            n_undetected   = int(np.sum(out[interpolation_flags == 0] != cd.scaled_quantized_padded_original[interpolation_flags == 0]))
            print(f'{l_scratch:<16} {n_erasures:<12} {n_interpolated:<14} {n_failed:<10} {n_undetected}')

        pass


if __name__ == "__main__":
    AudioCD.test()