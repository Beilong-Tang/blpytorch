import os

printed_dict={}

def dprint(msg:str, name=None, test = False):
    if test:
        if os.getenv('DPRINT_TEST') in [None, '0', 'true']:
            return
    if name is None:
        name = msg
    if name not in printed_dict:
        print(f"[DPRINT]: {msg}")
        printed_dict[name] = True
