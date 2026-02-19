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

        #insert your code here

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

        #insert your code here

        assert len(np.shape(output))==1 and type(output) is np.ndarray, 'output must be a 1D numpy array'
        return (output,n_frames)

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

        #insert your code here

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

        #insert your code here

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

        #insert your code here

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

        #insert your code here

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

        #insert your code here

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

        data = np.fromstring(sdata, dtype=typ)
        data=data/(2**15)
        ch_1 = data[0::nch]
        ch_2 = data[1::nch]
        audiofile=np.transpose(np.vstack((ch_1,ch_2)))
        cd = AudioCD(Fs,1,8)
        cd.writeCd(audiofile)
        T_scratch = 600000 # Scratch at a diameter of approx. 66 mm
        l_scratch = 3000
        for i  in range(math.floor((cd.cd_bits).size/T_scratch)):
            cd.scratchCd(l_scratch,30000+(i)*T_scratch)

        [out,interpolation_flags] = cd.readCd()
        cd.save_and_play_music(out[:,0],out[:,1],'test.wav',0)
        print('end')



        print(f'Number samples with erasure flags: {np.sum(interpolation_flags!=0)}')
        print(f'Number samples with failed interpolations: {np.sum(interpolation_flags==-1)}')
        print(f'Number undetected errors: {np.sum(out[interpolation_flags==0] != cd.scaled_quantized_padded_original[interpolation_flags==0])}')

        pass
