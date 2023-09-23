elif m is SimFusion_4in:
    c2 = sum(ch[x] for x in f)
elif m is SimFusion_3in:
    c2 = args[0]
    if c2 != no:  # if not output
        c2 = make_divisible(c2 * gw, 8)
    args = [[ch[f_] for f_ in f], c2]
elif m is IFM:
    c1 = ch[f]
    c2 = sum(args[0])
    args = [c1, *args]
elif m is InjectionMultiSum_Auto_pool:
    c1 = ch[f[0]]
    c2 = args[0]
    args = [c1, *args]
elif m is PyramidPoolAgg:
    c2 = args[0]
    args = [sum([ch[f_] for f_ in f]), *args]
elif m is AdvPoolFusion:
    c2 = sum(ch[x] for x in f)
elif m is TopBasicLayer:
    c2 = sum(args[1])