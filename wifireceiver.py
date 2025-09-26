# -*- coding: utf-8 -*-
from random import randint
import numpy as np
import sys
import commpy as comm
import commpy.channelcoding.convcode as check
from pip import main
import matplotlib.pyplot as plt


def WifiReceiver(input_stream, level):

    nfft = 64
    Interleave_tr = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[4,-1])),[-1,])
    preamble = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    cc1 = check.Trellis(np.array([3]),np.array([[0o7,0o5]]))

    # set zero padding to be 0, by default
    begin_zero_padding = 0
    message=""
    length=0

    if level >= 4:
        #Input QAM modulated + Encoded Bits + OFDM Symbols in a long stream
        #Output Detected Packet set of symbols
        input_stream=input_stream
        # awgn -> noise -> fft -> demod -> preamble -> deconv -> deinterleave -> length
        min_snr = 30

        #use preamble to identify length bits:
        preamble = bit2complex(preamble) # convert to complex
        preamble_ifft = np.fft.ifft(preamble) # get fftd preamble

        #find best match
        match = np.correlate(input_stream, preamble_ifft, mode='valid') # find closest match to preamble in input
        preamble_idx = np.argmax(np.abs(match)) #idx where preamble starts
        begin_zero_padding = preamble_idx

        input_stream = input_stream[(begin_zero_padding + len(preamble)):] #[message, end zero]
        input_stream = de_ifft(input_stream, nfft) #undo ifft
        length, message = calculate_length(input_stream, False, nfft)

        decoded_bits = soft_decode(message, cc1, length) # handles end zeros
        deint = deinterleave(decoded_bits, nfft)
        message = decode_message(deint)[:length]
        return begin_zero_padding, message, length

    if level >= 3:
        #Input QAM modulated + Encoded Bits + OFDM Symbols
        #Output QAM modulated + Encoded Bits
        # fft -> demod -> preamble -> deconv -> deinterleave -> length
        input_stream = input_stream
        input_stream = de_ifft(input_stream, nfft) #undo ifft
        input_stream = input_stream[len(preamble) // 2:] # remove preamble
        length, message = calculate_length(input_stream, False, nfft) # separate length and message
        decoded_bits = soft_decode(message, cc1, length)
        deint = deinterleave(decoded_bits, nfft)
        message = decode_message(deint)[:length]
        return begin_zero_padding, message, length
    
    if level >= 2:
        #Input QAM modulated + Encoded Bits
        #Output Interleaved bits + Encoded Length
        input_stream=input_stream
        # demod -> preamble -> deconv -> deinterleave -> length
        input_stream = input_stream[len(preamble) // 2:] # remove preamble
        length, message = calculate_length(input_stream, False, nfft) # separate length and message
        #print(f"before decoding: {message}, len {len(message)}")
        decoded_bits = soft_decode(message, cc1, length)
        #print(f"after decoding: {decoded_bits}, length {len(decoded_bits)}")
        deint = deinterleave(decoded_bits, nfft)
        message = decode_message(deint)[:length]
        return begin_zero_padding, message, length
       
    if level >= 1:
        #Input Interleaved bits + Encoded Length
        #Output Deinterleaved bits
        input_stream = input_stream
        length, message = calculate_length(input_stream, True, nfft)
        deint = deinterleave(message, nfft)
        message = decode_message(deint)[:length]
        return begin_zero_padding, message, length

    raise Exception("Error: Unsupported level")

# decode length of message at head of input
# binary: true / false for non-mod
def calculate_length(input, binary, nfft):
    length_bits = input[0:2*nfft] if binary else input[0:nfft] # always 128 bits long
    decoded_len = []

    # demodulate if complex
    if not binary:
        mod = comm.modulation.QAMModem(4)
        length_bits = mod.demodulate(length_bits, demod_type = 'hard') # convert to binary

    #extract every three bits and take majority
    for i in range(0, len(length_bits), 3):
        mode = 1 if (np.sum(length_bits[i:i+3]) >= 2) else  0
        decoded_len.append(mode)

    # sanity check
    if len(decoded_len) > (np.round(nfft*2 / 3)):
        raise Exception(f"Error: decoded length too long: {len(decoded_len)}")
    
    #print(f"decoded length: {decoded_len}")
    length = comm.utilities.bitarray2dec(decoded_len) # actual decimal length of message
    message = np.array(input[2*nfft:]) if binary else np.array(input[nfft:])# get message portion from input
    #print(f"just the message: {len(message)}")
    return length, message

def bit2complex(bits):
    complex_map = { # map decimal to cc according to QAM4
        0: -1 - 1j,
        1: -1 + 1j,
        2: 1 - 1j,
        3: 1 + 1j 
    }
    complex = []
    for i in range(0, len(bits), 2):
        pair = bits[i:i+2]
        decimal = pair[0]*2 + pair[1]
        complex.append(complex_map[decimal])

    return np.array(complex)


def deinterleave(bits, nfft):
    interleave = np.reshape(np.transpose(np.reshape(np.arange(0, 2*nfft, 1),[-1,4])),[-1,])

    deinterleave_seq = np.zeros(nfft*2, dtype = int)
    deinterleave_seq[interleave] = np.arange(2*nfft)

    chunk_num = int(len(bits)/(2*nfft))
    output = np.zeros(shape=(len(bits),))
    for i in range(chunk_num):
        chunk = bits[i*2*nfft:(i+1)*2*nfft]
        output[i*2*nfft:(i+1)*2*nfft] = chunk[deinterleave_seq]

    return output

# soft Viterbi decoding
def soft_decode(message, cc1, length):
    next_table = cc1.next_state_table # next stages
    output_table = cc1.output_table
    num_states = cc1.number_states
    num_inputs = cc1.number_inputs
    complex_map = { # map decimal to cc according to QAM4
        0: -1 - 1j,
        1: -1 + 1j,
        2: 1 - 1j,
        3: 1 + 1j 
    }
    #print(f"next table: {next_table}")
    #print(f"output table: {output_table}")

    path_distances = np.full(num_states, np.inf) # column represents distance to states
    path_distances[0] = 0
    path_inputs = [] # (i x [nums_tates]): corresponding inputs of each best path to states at each layer
    path_states = [] #(i x [num_states]) corresponding states of each path to each states at each layer

    for i in message:
        #print(f"bit: {i}\n")
        # num states x num input array: next states x 2 possible paths to the next state
        # each element is [curr state, input, distance]
        temp_paths = [[] for _ in range(num_states)]

        # build up path routes for each next state
        for curr_state in range(num_states): # determine distances to next paths
            for input in range(num_inputs): # determine possible paths based on input
                next_state = next_table[curr_state][input]
                output = output_table[curr_state][input]
                out_complex = complex_map[output]
                dist = np.abs(i - out_complex) ** 2 #calculate local distance
                full_dist = path_distances[curr_state] + dist # calculate total distance
                #print(f"state{curr_state} -> state{next_state} by {input}: {dist}\n")
                temp_paths[next_state].append([curr_state, input, full_dist]) # add as a path

        #print(f"Possible paths: {temp_paths}\n")

        # choose best path for each state
        temp_states = np.full(num_states, -1) # (1xnum_states)next states at each bit 
        temp_inputs = np.full(num_states, -1) # 1 x num_states input to next state at each bit
        for next_state in range(num_states):
            choices = temp_paths[next_state] # gets row, 2 elements
            best_path = choices[0] if (choices[0][2] <= choices[1][2]) else choices[1] # get smallest total distance to state
            path_distances[next_state] = best_path[2] #add smallest distance
            temp_states[next_state] = best_path[0]
            temp_inputs[next_state] = best_path[1]
        path_states.append(temp_states) # map previous states to next state
        path_inputs.append(temp_inputs) # map inputs to next states

        #print(f"updated path: {path_distances}\n")
        #print(f"updated states: {path_states}\n")
        #print(f"updated inputs: {path_inputs}\n")

    # back track to get shortest path:
    decoded = []
    best_path_end = np.argmin(path_distances)
    curr_state = best_path_end # states to shortest end state
    for i in reversed(range(len(path_states))): 
        bit = path_inputs[i][curr_state] # get input to that state
        curr_state = path_states[i][curr_state] # move to previous state
        decoded.insert(0, bit)

    return np.array(decoded, dtype = int)

# convert back to letters
# input: deinterleaved bits
def decode_message(decoded):
    end_zero = len(decoded) % 8 # end zero padding
    decoded = decoded[:-end_zero] if end_zero > 0 else decoded
    bytes = [np.array(decoded[i:i+8], dtype = int) for i in range(0, len(decoded),8)]
    decimals = [comm.utilities.bitarray2dec(byte) for byte in bytes]
    message = ''.join(chr(i) for i in decimals)
    return message

# undo ifft
def de_ifft(input, nfft):
    nsym = int(len(input)/nfft) 
    for i in range(nsym):
        symbol = input[i*nfft:(i+1)*nfft]
        # de-ifft
        input[i*nfft:(i+1)*nfft] = np.fft.fft(symbol)
    
    return input
    

# for testing purpose
from wifitransmitter import WifiTransmitter
if __name__ == "__main__":
    test_case = 'After all, I absorb the world around me, and that’s changing all the time. Just as all the water that was in my body last time we met has now been replaced with new water, the things that make up me have changed too.'
    symbols = [randint(0, 1) for i in range(32*8)]
    
    # test level 1: bit interleaving
    output1 = WifiTransmitter(test_case, 1)
    begin_zero_padding1, message1, length1 = WifiReceiver(output1, 1)
    print(begin_zero_padding1, repr(message1), length1)
    print(repr(test_case) == repr(message1))

    # test level 2: Turbo coding & 4-QAM modulation
    output2 = WifiTransmitter(test_case, 2)
    begin_zero_padding2, message2, length2 = WifiReceiver(output2, 2)
    print(begin_zero_padding2, message2, length2)
    print(repr(test_case) == repr(message2))    
   
    # test level 3: OFDM encoding
    output3 = WifiTransmitter(test_case, 3)
    begin_zero_padding3, message3, length3 = WifiReceiver(output3, 3)
    print(begin_zero_padding3, message3, length3)
    print(test_case == message3)
    
    # test level 4:Noise addition & zero padding
    noise_pad_begin, output4, length = WifiTransmitter(test_case, 4, -5)
    print(f"Noise: {noise_pad_begin}, len: {length}")
    begin_zero_padding4, message4, length4 = WifiReceiver(output4, 4)
    print(begin_zero_padding4, message4, length4)
    print(test_case == message4)
    