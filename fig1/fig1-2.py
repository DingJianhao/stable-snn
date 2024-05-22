import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))

def lif_spike(input, mem):
    mem = 0.9 * mem + input
    s = float(mem > 1)
    mem2 = mem - s * mem
    return s, mem, mem2

def lif_no_spike(input, mem):
    mem = 0.9 * mem + input
    s = 0 #float(mem > 1)
    mem2 = mem - s * mem
    return s, mem, mem2

record_mem1 = []
mem1_postspk = 0
record_mem1.append((0,0))

record_mem2 = []
mem2_postspk = 0
record_mem2.append((0,0))

record_mem3 = []
mem3_postspk = 0
record_mem3.append((0,0))

perturbs = [0]
avg_input_difference = [0]
delta_spike = [0]
spk_cnt1 = 0
spk_cnt2 = 0
s1 = 0
s2 = 0
delta_spk = 0
for t in range(0, 30):
    delta_spk += np.abs(s1 - s2)
    avg_input_difference.append((spk_cnt2 - spk_cnt1)/30)
    delta_spike.append(delta_spk)

    s1, mem1_prespk, mem1_postspk = lif_spike(0.3, mem1_postspk)
    record_mem1.append((t+0.98, mem1_prespk))
    if s1 > 0:
        record_mem1.append((t+0.99, mem1_postspk))
    else:
        record_mem1.append((t + 0.99, mem1_prespk))


    rnd = np.random.randn(1)[0] * 0.1
    s2, mem2_prespk, mem2_postspk = lif_spike(0.3 + rnd, mem2_postspk)
    record_mem2.append((t + 0.98, mem2_prespk))
    if s2 > 0:
        record_mem2.append((t + 0.99, mem2_postspk))
    else:
        record_mem2.append((t + 0.99, mem2_prespk))

    s3, mem3_prespk, mem3_postspk = lif_no_spike(rnd, mem3_postspk)
    if s3 > 0:
        record_mem3.append((t + 0.99, mem3_postspk))
    else:
        record_mem3.append((t + 0.99, mem3_prespk))

    spk_cnt2 += s2
    spk_cnt1 += s1

    perturbs.append(rnd)



print(avg_input_difference)
print(delta_spike)


# print(record_mem1)
# print(record_mem2)

record_mem2 = np.array(record_mem2)
record_mem1 = np.array(record_mem1)
record_mem3 = np.array(record_mem3)

plt.plot(record_mem1[:,0], record_mem1[:,1], color='tomato', alpha=1, label='V before perturbation')
plt.plot(record_mem2[:,0], record_mem2[:,1], color='deepskyblue', alpha=1, label='V after perturbation')

plt.step(np.arange(0,31), perturbs, color='k', label='Perturbation')
plt.step(np.arange(0,31), np.array(avg_input_difference), color='teal', label=r'TASAD$^2$')
plt.step(np.arange(0,31), np.array(delta_spike)/30, color='purple', label='STD$^2$/T')
plt.plot(record_mem3[:,0], record_mem3[:,1], color='orange', label='MPP')

plt.fill_between(record_mem1[:,0],0, record_mem1[:,1],color='tomato', alpha=0.4)
plt.fill_between(record_mem2[:,0],0, record_mem2[:,1],color='deepskyblue', alpha=0.4)




plt.ylabel('Membrane potential',fontsize=20)
plt.xlabel('Time step',fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.ylim(-0.5, 3)
plt.legend(ncol=1,fontsize=16,frameon=False,loc='upper left', handlelength=1)
plt.subplots_adjust(bottom=0.16,top=0.95,left=0.15,right=0.98)

plt.savefig('potential2.pdf')

plt.show()