from plumbum import local
from plumbum.cmd import xmds2, python3

chain1 = xmds2['transport.xmds']
chain1()

chain2 = local['./bec_transport']
chain2()

chain3 = python3['output_transport.py']
chain3()