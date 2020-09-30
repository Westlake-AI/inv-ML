# Coil-20: ML-Enc baseline + Extra-Head + Orth

def import_test_config(test_num, mode='encoder'):
    if test_num == 1:
        return test_1(mode)
    # elif test_num == 2:
    #     return test_2(mode)
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
                layer            = [4096, 4096, 4096, 4096,   4096, 4096, 4096,  4096,   10],
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
                layer            = [4096, 4096, 4096, 4096,   4096, 4096, 4096,  4096,   10],
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
                extra_w        = [0,   0,    0],
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




# coil-20, 6 layers Extra + Orth loss + Padding, (classs=20), d=2, ok
def test_1(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            BATCHSIZE = 1440,
            N_dataset = 1440,
            # EPOCHS = 9000,
            EPOCHS = 12000,
            regularB = 2.5,
            MAEK = 5,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=6,
                        orth=0.1,
                        pad=10
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   2],
                relu             = [     1,    1,    1,    1,       0],
                Enc_require_gard = [     1,    1,    1,    1,       1],
                Dec_require_gard = [     0,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     20,   10,    6,      2,    ],
                weight           = [       2,    8,    32,     64,   ],
                push_w           = [     5e-1,   1,     8,     18,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     1],
                dec_forward      = [0,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   30],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    3],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [4000, 10000,  0],
                    enc_w   = [3500, 11000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 9500,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [ 9500,  7000,  3500,    900,   ],
            ),
            # inverse mode, 0815-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  4080, 4040, 4000, 3900, 3800], # 0815-1
                pad_w            =  [     0,    2,    4,    8,    16,   32],
                pad_gradual = [0,  2500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            MAEK = 15,
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   2],
                relu             = [    1,    1,    1,    1,       0],
                Enc_require_gard = [    0,    0,    0,    0,       0],
                Dec_require_gard = [    1,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     0],
                dec_forward      = [1,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,  0,  0],
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


# coil-20, 6 layers Extra + Orth loss + Padding, (classs=20), d=2
def test_3(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            BATCHSIZE = 1440,
            N_dataset = 1440,
            # EPOCHS = 12000,
            EPOCHS = 15000,
            regularB = 2.5,
            MAEK = 5,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=6,
                        orth=0.1,
                        pad=10
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   2],
                relu             = [     1,    1,    1,    1,       0],
                Enc_require_gard = [     1,    1,    1,    1,       1],
                Dec_require_gard = [     0,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     20,   10,    6,      2,    ],
                weight           = [       2,    8,    32,     64,   ],
                push_w           = [     5e-1,   1,     8,     18,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     1],
                dec_forward      = [0,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   30],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    3],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3000, 14000,  0],
                    enc_w   = [2500, 13500,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [1500, 13000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [ 9500,  7000,  3500,    900,   ],
            ),
            # inverse mode, 0815-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  4080, 4040, 4000, 3900, 3800],
                pad_w            =  [     0,    2,    4,    8,    16,   32],
                # pad_gradual = [0,  2500, 1],
                pad_gradual = [0,  3000, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            MAEK = 15,
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   2],
                relu             = [    1,    1,    1,    1,       0],
                Enc_require_gard = [    0,    0,    0,    0,       0],
                Dec_require_gard = [    1,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     0],
                dec_forward      = [1,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,  0,  0],
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


# coil-20, 6 layers Extra + Orth loss + Padding, (classs=20), d=20
def test_4(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS = 12000,
            # EPOCHS = 15000,
            regularB = 2.2,
            MAEK = 5,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=6,
                        orth=0.1,
                        pad=10
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   20],
                relu             = [     1,    1,    1,    1,       0],
                Enc_require_gard = [     1,    1,    1,    1,       1],
                Dec_require_gard = [     0,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                # layer            = [0,     20,   10,    6,      2,    ],
                layer            = [0,    40,   30,    20,     10,    ],
                weight           = [       2,    8,    32,     64,   ],
                push_w           = [     5e-1,   1,     8,     18,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     1],
                dec_forward      = [0,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   30],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    3],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3000, 14000,  0],
                    enc_w   = [2500, 13500,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [1500, 13000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                # each             =   [ 9500,  7000,  3600,    1500,   ], # OK
                each             =   [ 9600,  8000,  3600,    1600,   ],
            ),
            # inverse mode, 0815-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  4080, 4040, 4000, 3900, 3800],
                pad_w            =  [     0,    2,    4,    8,    16,   32],
                # pad_gradual = [0,  2500, 1],
                pad_gradual = [0,  3000, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            MAEK = 15,
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,   20],
                relu             = [    1,    1,    1,    1,       0],
                Enc_require_gard = [    0,    0,    0,    0,       0],
                Dec_require_gard = [    1,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     0],
                dec_forward      = [1,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,  0,  0],
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


# coil-20, 6 layers Extra + Orth loss + Padding, (classs=20), d=2,
def test_5(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS = 12000,
            # EPOCHS = 15000,
            regularB = 2.2,
            MAEK = 5,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=6,
                        orth=0.1,
                        pad=10
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,    2],
                relu             = [     1,    1,    1,    1,       0],
                Enc_require_gard = [     1,    1,    1,    1,       1],
                Dec_require_gard = [     0,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     10,   6,     4,      2,   ],
                weight           = [       2,    8,    32,     64,   ],
                push_w           = [     5e-1,   1,     8,     18,   ],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     1],
                dec_forward      = [0,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,   30],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    3],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3000, 14000,  0],
                    enc_w   = [2500, 13500,  0],
                    dec_w   = [0,    0,      0],
                    each_w  = [0,    0,      0],
                    extra_w = [1500, 13000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                # each             =   [ 9500,  7500,  3600,    1600,   ], # OK
                each             =   [ 9800,  8200,  3600,    1600,   ],
            ),
            # inverse mode, 0815-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  4080, 4040, 4000, 3900, 3800],
                pad_w            =  [     0,    2,    4,    8,    16,   32],
                # pad_gradual = [0,  2500, 1],
                pad_gradual = [0,  3000, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            MAEK = 15,
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [ 4096, 4096, 4096, 4096,  4096,    2],
                relu             = [    1,    1,    1,    1,       0],
                Enc_require_gard = [    0,    0,    0,    0,       0],
                Dec_require_gard = [    1,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,],
                enc_forward      = [0,    0,    0,    0,      0,     0],
                dec_forward      = [1,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,  0,  0],
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
