# mnist baseline + ExtraHead + Orth Loss

def import_test_config(test_num, mode='encoder'):
    if test_num == 1:
        return test_1(mode)
    elif test_num == 2:
        return test_2(mode)
    elif test_num ==3:
        return test_3(mode)
    elif test_num ==4:
        return test_4(mode)
    elif test_num ==5:
        return test_5(mode)
    # elif test_num ==6:
    #     return test_6(mode)
    # elif test_num ==7:
    #     return test_7(mode)
    # elif test_num ==8:
    #     return test_8(mode)


def base_params(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            EPOCHS=8000,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8, orth=0, pad=0),
            add_jump = True,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [],
                weight           = [],
            ),
            # AE
            AEWeight = dict(
                each             = [],
                AE_gradual = [0,  0,  1],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,    1],
                enc_forward_w  = [1,   1,    1], 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual   = [0,   0,    1],
                each           = [],
            ),
            ### inverse mode
            InverseMode = dict(
                mode = "pinverse", #"CSinverse",
                loss_type = "L2",
                padding          = [    0,    0,    0,   0,      0,     0,    0,    0],
                pad_w            = [    0,    0,    0,   0,      0,     0,    0,    0],
                pad_gradual = [0,   0,   1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            EPOCHS= 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            add_jump = True,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [],
                weight           = [],
            ),
            # AE
            AEWeight = dict(
                each             = [],
                AE_gradual = [0,  0,  1],
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [0,   0,    0],
                enc_forward_w  = [0,   0,    0], 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # [1 -> 0]
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# mnist, 8layers ML-Enc + ExtraHead + Orth, d=2, good
def test_1(mode):
    # Source: mnist, 8 layers inverse + DR + padding, (classs=10), 0819-8
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            # BATCHSIZE = 8000,
            # N_dataset = 8000,
            # EPOCHS = 10000, # ori
            EPOCHS = 12000,
            regularB = 3,
            MAEK = 15, # 0819-8
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        # push=0.8,
                        push=5,
                        orth=0.1,
                        # pad=100 # 0814-6
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0818
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ], # 0819
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ], # 0819
                push_w           = [     1e-1,  5e-1,  1,       4,    8,   16,   22,   ], # Good
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2],
                # gradual
                LIS_gradual = [0,    0,     1],
                push_gradual = dict(
                    cross_w = [4000, 12000,  0],
                    enc_w   = [3000, 11000,  0],
                    dec_w   = [0,     0,     0],
                    each_w  = [0,     0,     0],
                    extra_w = [2500,  9500,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8800,  6100,  2400,  1200,   700,  350,  100,   ],  # 0820-1-3, try
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0818
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8layers ML-Enc + ExtraHead + Orth,
def test_2(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            # BATCHSIZE = 10000,
            # N_dataset = 10000,
            BATCHSIZE = 8000,
            N_dataset = 8000,
            EPOCHS = 10000,
            regularB = 3,
            # MAEK = 30,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=5,
                        orth=0.1,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # try ori
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0819-8, ori
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     4,    8,   10,      12,   10,    5,    2,   ], # 0819-8
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ], # 0819-8
                push_w           = [     2e-1,   1,    4,       8,   18,   32,   10,   ], # 0820-3-2, Plan B
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3500, 10000, 0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5800,  2600,  1200,    800,  400,  100,   ],  # 0820-3-2
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # ori
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0819,
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1], # 0730, add new
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8layers ML-Enc + ExtraHead + Orth, d=2, ok inverse
def test_3(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            # BATCHSIZE = 10000,
            # N_dataset = 10000,
            BATCHSIZE = 8000,
            N_dataset = 8000,
            EPOCHS=10000,
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=5,
                        orth=0.1,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     4,    8,   10,      12,   10,    5,    2,   ], # 0819-8
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ], # 0819-8
                push_w           = [     2e-1,   1,    4,       8,   16,   32,   11,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2], # 0819-8
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3500, 10000, 0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [9500,  5800,  2100,  1000,    700,  400,  100,   ],
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,  # 0812
            DATASET="mnist",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # ori
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0819,
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,  1,  1],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8layers ML-Enc + ExtraHead + Orth, d=2, ok
def test_4(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            # BATCHSIZE = 10000,
            # N_dataset = 10000,
            BATCHSIZE = 8000,
            N_dataset = 8000,
            EPOCHS=10000,
            # regularB = 100,
            regularB = 3,
            MAEK = 15, # 0819-8
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        # push=0.8,
                        push=5,
                        # push=200,
                        orth=0.1,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # try ori
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0819-8,
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     4,    8,   10,      12,   10,    5,    2,   ], # 0819-8
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ], # 0819-8
                push_w           = [     2e-1,   1,    4,       8,   16,   30,   12,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3500, 10000, 0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [9200,  6000,  2000,  1000,    700,  400,  100,   ], # 0820-6
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,  # 0812
            DATASET="mnist",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # ori
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0819, ori
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push]
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8layers ML-Enc + ExtraHead + Orth, d=10
def test_5(mode):
    # copy of 0812: 8 layers inverse + DR + padding, (classs=10), (0812-2)
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS = 10000,
            regularB = 3,
            MAEK = 20,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=1,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     12,   10,   8,      6,    4,    3,     2,  ],
                weight           = [       2,    4,    8,     16,   32,   64,   128,  ],
                push_w           = [      2e-1,  1,    2,      4,    8,   16,    24,  ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [4000, 9000,  0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [3000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5500,  2000,  800,    500,  300,  100,   ],
            ),
            # inverse mode
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000, 
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push]
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,  1,  1], 
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8 layers ExtraHead + Orth, (classs=10), d=10, ok
def test_6(mode):
    # copy of 0812: 8 layers inverse + DR + padding, (classs=10), (0812-2)
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS = 10000,
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        # push=0.8,
                        push=2,
                        orth=0.2,
                        pad=0
                        # pad=100 # 0814-6
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     12,   10,   8,      6,    4,    3,     2,  ],
                weight           = [       2,    4,    8,     16,   32,   64,   128,  ],
                # push_w           = [      2e-1,  1,    2,      4,    8,   16,    24,  ],
                push_w           = [      2e-1,  1,    8,     16,    8,   32,    12,  ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [4000, 9000,  0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [3000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5500,  2000,  800,    500,  300,  100,   ],
            ),
            # inverse mode,
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000, 
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push]
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,  1,  1],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params


# mnist, 8 layers ExtraHead + Orth, (classs=10), d=2, ok
def test_7(mode):
    # Source: mnist, 8 layers inverse + DR + padding, (classs=10), 0819-8
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            # BATCHSIZE = 9000,
            # N_dataset = 9000,
            EPOCHS = 15000,
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=5,
                        orth=0.1,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0818
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ], # 0819
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ], # 0819
                push_w           = [     1e-1,  5e-1,  1,       4,    8,   16,   22,   ], # Good
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2],
                # gradual
                LIS_gradual = [0,    0,     1],
                # push_gradual = dict( # good, 0918
                #     cross_w = [4000, 12000,  0],
                #     enc_w   = [3000, 11000,  0],
                #     dec_w   = [0,     0,     0],
                #     each_w  = [0,     0,     0],
                #     extra_w = [2500,  9500,  0],
                # ),
                push_gradual = dict(
                    cross_w = [4000, 14000,  0],
                    enc_w   = [3000, 13000,  0],
                    dec_w   = [0,     0,     0],
                    each_w  = [0,     0,     0],
                    extra_w = [2500, 11500,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8800,  6100,  2400,  1200,   700,  350,  100,   ],
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2], # 0818
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push]
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    params.update(param)
    return params

