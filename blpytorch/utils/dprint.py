printed_dict={}

def dprint(msg:str, name=None):
    if name is None:
        name = msg
    if name not in printed_dict:
        print(f"[DPRINT]: {msg}")
        printed_dict[name] = True