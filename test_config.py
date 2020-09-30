# config for main test, 2020.0928

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
    elif test_num ==6:
        return test_6(mode)
    elif test_num ==7:
        return test_7(mode)
    elif test_num ==8:
        return test_8(mode)
    else:
        return None


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
                extra_w        = [1,   1,    1], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0],
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


# mnist, 8layers ML-Enc + ExtraHead + Orth, d=2
def test_1(mode):
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
            EPOCHS = 12000,
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
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ],
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ],
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
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
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


# mnist, 8layers ML-Enc + ExtraHead + Orth + Padding, d=2
def test_2(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            # BATCHSIZE = 20000,
            # N_dataset = 20000,
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=10000,
            regularB = 100,
            # regularB = 10,
            # MAEK = 30,
            MAEK = 20,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=100
                    ),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     12,   10,   8,      6,    4,    3,    2,    ],
                weight           = [       2,    4,    8,      16,   32,   64,  128,   ],
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
                LIS_gradual = [0,    0,     1],
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
                each             =   [8000,  5500,  1500,  600,    350,  200,  100,   ],
            ),
            # inverse mode
            InverseMode = dict(
                mode = "ZeroPadding", #"CSinverse",
                loss_type = "L1",
                padding          =  [     0,  780, 760, 700,   750,  650,  700, 630,  600],
                pad_w            =  [     0,    2,   4,   16,     8,   32,   16,  64,  64],
                pad_gradual = [0,  3500, 1],
            ),
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
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
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
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0], # [dist, angle, push]
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


# usps, 8 layers Extra + Orth + padding, d=2
def test_3(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="usps",
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=10000,
            regularB = 10,
            MAEK = 20,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=10
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,    2],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     12,   10,   8,      6,    4,    3,    2,    ],
                weight           = [       2,    4,    8,      16,   32,   64,  128,   ],
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
                    cross_w = [4000, 9000,  0], # 0801-1, ok
                    enc_w   = [3000, 8000,  0], # 0809-1, try
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [3000, 8000,  0], # 0809-1, try
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5500,  1500,  600,    350,  200,  100,   ],
            ),
            # inverse mode, 0808-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  255, 250, 250,   240,  230, 220, 210, 200],
                pad_w            =  [     0,    2,   4,   8,     16,  32,   64,  128,  256],
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="usps",
            BATCHSIZE = 9298, 
            N_dataset = 9298,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,    2],
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


# Kmnist, 8layers ML-Enc + ExtraHead + Orth, d=2,
def test_4(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="Kmnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
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
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ],
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ],
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
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="Kmnist",
            BATCHSIZE = 10000, 
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                # relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
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
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1],
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


# fmnist, 8layers ExtraHead + Orth + Padding, d=2
def test_5(mode):
    # Source: mnist, 8 layers inverse + DR + padding, (classs=10), 0819
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET = "Fmnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=12000,
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        push=5,
                        orth=0.1,
                        pad=20
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
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ],
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ],
                push_w           = [     1e-1,  5e-1,  1,       4,    8,   16,   28,   ],
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
                cross_w        = [1,   1,   15],
                enc_forward_w  = [1,   1,   30],
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
                each             =   [8900,  6800,  2500,  1200,   700,  350,  100,   ],
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET = "Fmnist",
            BATCHSIZE = 10000, 
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [784,  1000, 1000, 1000,   1000,  1000, 1000, 1000,   2],
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
                # [dist, angle, push],
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


# coil-20, 6 layers Extra + Orth + Padding, (classs=20), d=2
def test_6(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 20,
            DATASET="coil-20",
            BATCHSIZE = 1440,
            N_dataset = 1440,
            EPOCHS = 15000,
            regularB = 2.5,
            MAEK = 5,
            PlotForloop = 500,
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
                cross = [1,],
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


# coil-100, 5 layers Extra + Orth, (classs=100), d=100
def test_7(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 100,
            DATASET="coil-100",
            BATCHSIZE = 7200,
            N_dataset = 7200,
            EPOCHS = 10000,
            regularB = 2.5,
            MAEK = 8, # 10
            PlotForloop = 500,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.5,
                        pad=0
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [ 3072, 3072, 3072, 3072,  3072,  100],
                relu             = [     1,    1,    1,    1,       0],
                Enc_require_gard = [     1,    1,    1,    1,       1],
                Dec_require_gard = [     0,    0,    0,    0,       0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     80,   60,   30,     10,    ],
                weight           = [       2,    8,    32,     64,    ],
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
                cross_w        = [1,   1,   10],
                enc_forward_w  = [1,   1,   10],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    1], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [2000, 8000,   0],
                    enc_w   = [3500, 10000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [3000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [ 9500,  7500,  3200,    600,   ],
            ),
            # inverse mode,
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 100,
            DATASET="coil-100",
            MAEK = 15,
            BATCHSIZE = 7200,
            N_dataset = 7200,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [ 3072, 3072, 3072, 3072,  3072,  100],
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
                # [dist, angle, push]
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


# Cifar-10, 8 layers inverse + DR + padding, d=2, bad DR
def test_8(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET = "cifar-10",
            BATCHSIZE = 8000,
            N_dataset = 8000,
            EPOCHS=12000,
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
                layer            = [3072, 3500, 3500, 3500,   3500,  3500, 3500, 3500,   2],
                relu             = [    0,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,    80,   40,   25,      10,    8,    4,    2,   ],
                weight           = [       1,    2,    4,       8,   16,   32,   64,   ],
                push_w           = [     1e-1,  5e-1,  1,       4,    8,   16,   22,   ], # 0820-1-3,Good!, try cifar10
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
                # cross_w        = [1,   1,   10], # ori
                # enc_forward_w  = [1,   1,   10], # ori
                cross_w        = [1,   1,   15], # 0821-1-1, try
                enc_forward_w  = [1,   1,   30], # 0822-7-1, try
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   0,    2],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # 0820-1, try
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
                each             =   [9000,  7000,  2600,  1200,   800,  350,  100,   ],  # 0823-7-1, try
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET = "cifar-10",
            BATCHSIZE = 8000, 
            N_dataset = 8000,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [3072, 3500, 3500, 3500,   3500,  3500, 3500, 3500,   2],
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
                # [dist, angle, push],
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
