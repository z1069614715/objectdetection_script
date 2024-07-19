if type(x) in {list, tuple}:
    if idx == (len(self.model) - 1):
        if type(x[1]) is dict:
            print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x[1]["one2one"]])}')
        else:
            print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x[1]])}')
    else:
        print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x if x_ is not None])}')
elif type(x) is dict:
    print(f'layer id:{idx:>2} {m.type:>50} output shape:{", ".join([str(x_.size()) for x_ in x["one2one"]])}')
else:
    if not hasattr(m, 'backbone'):
        print(f'layer id:{idx:>2} {m.type:>50} output shape:{x.size()}')