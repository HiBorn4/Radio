# Import the random module.
import random
import numpy as np
import matplotlib.pyplot as plt

# jamming_rate = int(input("Enter Jamming Rate: "))
# send_rate = int(input("Enter Sender Rate: "))
# time_interval = int(input("Enter Time Interval: "))

jamming_rate = 2  
send_rate = 1  
time_interval = 100 


def generate_signals(rate, num_cycle):
  signals = np.array([])
  for i in range(0, num_cycle, rate):
    curr_bit = np.random.randint(low=0,high=2)

    if i + rate > num_cycle:
      repeat_time = num_cycle - i
    else:
      repeat_time = rate

    for i in range(repeat_time):
      signals = np.append(signals, curr_bit)
  return signals

def sum_bits(sender_bits, jammer_bits):

    def sum_bits_helper(sender_bits, jammer_bits, start, end):
        
        if start == end:
            return np.array([int(sender_bits[start]) + int(jammer_bits[start])])

        mid = (start + end) // 2

        left_result = sum_bits_helper(sender_bits, jammer_bits, start, mid)

        right_result = sum_bits_helper(sender_bits, jammer_bits, mid + 1, end)

        return np.concatenate((left_result, right_result))

    return sum_bits_helper(sender_bits, jammer_bits, 0, len(sender_bits) - 1)


def filter_jammer_bits(result_bits):
    filtered_bit = np.array([]) 

    for i in range(len(result_bits)):
        if result_bits[i] == 2: 
            filtered_bit = np.append(filtered_bit, 1)  
        elif result_bits[i] == 0:  
            filtered_bit = np.append(filtered_bit, 0)  
        elif result_bits[i] == 1:  
            if result_bits[i-1] == 2:
                filtered_bit = np.append(filtered_bit, 0) 
            elif result_bits[i-1] == 0:  
                filtered_bit = np.append(filtered_bit, 1)  
            else: 
                filtered_bit = np.append(filtered_bit, 3) 

    return filtered_bit  


jammer_signal = generate_signals(jamming_rate, time_interval)
sender_signal = generate_signals(send_rate, time_interval)
mixed_signal = sum_bits(sender_signal, jammer_signal)
processed_signal = filter_jammer_bits(mixed_signal)

print("Sender Signals:", sender_signal)

print("Filtered Signals:", processed_signal)

def draw_summary(jammer_signal, sender_signal, mixed_signal, processed_signal, time_interval):

    fig, axs = plt.subplots(3, figsize=(20, 6))
    fig.suptitle(f'{time_interval} Cycles - Result')

    t = np.linspace(start=0, stop=time_interval, num=time_interval, dtype=int)

    axs[0].title.set_text("Mixed signals")
    axs[0].plot(t, mixed_signal, label="mixed")
    axs[0].plot(t, jammer_signal, label="jammer")
    axs[0].plot(t, sender_signal, label="sender")

    axs[1].title.set_text("Jammer vs Sender")
    axs[1].plot(t, jammer_signal, label="jammer")
    axs[1].plot(t, sender_signal, label="sender")

    axs[2].title.set_text("Sender vs Approximation")
    overlapping = 0.30
    axs[2].plot(t, sender_signal, label="sender", c='red', alpha=overlapping, lw=5)
    axs[2].plot(t, processed_signal, label="inference", c='green', alpha=overlapping)

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

draw_summary(jammer_signal, sender_signal, mixed_signal, processed_signal, time_interval)

def numerical_summary(jammer_signal, sender_signal, mixed_signal, processed_signal, time_interval, jamming_rate, send_rate):
    accuracy = np.sum(np.equal(sender_signal, processed_signal)) * 100 / time_interval

    print(f"{time_interval} cycles - Jamming rate: {jamming_rate} - Send rate: {send_rate} - Accuracy: {accuracy}%")

numerical_summary(jammer_signal, sender_signal, mixed_signal, processed_signal, time_interval, jamming_rate, send_rate)

def bulk_test(test_list, jammer_rates, draw=False):
    for time_interval in test_list:
        for jamming_rate in jammer_rates:
            jammer_signals = generate_signals(jamming_rate, time_interval)
            sender_signal = generate_signals(send_rate, time_interval)

            received_signals = jammer_signals + sender_signal

            processed_signals = filter_jammer_bits(received_signals)

            numerical_summary(jammer_signals, sender_signal, received_signals,
                              processed_signals, time_interval, jamming_rate, send_rate)

            if draw:
                draw_summary(jammer_signals, sender_signal,
                             received_signals, processed_signals, time_interval)


bulk_test([10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5], [1, 2, 3, 4, 5])