# mnist baseline: AE, ML-Enc, ML-AE

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


# AE, 8layers baseline for MNIST, d=20
def test_1(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET="mnist",
            # BATCHSIZE = 20000, # ori
            # N_dataset = 20000,
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=10000,
            regularB = 100,
            MAEK = 40,
            PlotForloop = 1000,
            ratio = dict(AE=1,
                        dist=0, angle=0, push=0, orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   20],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            # ReluType = dict(
            #     type = "Leaky", # "InvLeaky"
            #     Enc_alpha = 0.1,
            #     Dec_alpha = 0.1,
            # ),
            # Extra Head (DR project)
            # LIS
            LISWeght = dict(
                cross = [0,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,    1],
                enc_forward_w  = [1,   0,    1],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [0,    0,     0], # 0801-1, OK
                    enc_w   = [0,    0,     0], # 0801-1, OK
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0], # 0801-1, OK
                ),
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS= 0, # 2000,
            PlotForloop = 2000,
            ratio = dict(AE=0.05, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   20],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
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
                extra_w        = [0,  0,  0], # 0730, add new
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


# ML-Enc, 8layers baseline for MNIST, d=10
def test_2(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            # BATCHSIZE = 20000,
            # N_dataset = 20000,
            BATCHSIZE = 10000, # 0806-1
            N_dataset = 10000,
            EPOCHS=4000,
            # regularB = 100,
            # MAEK = 40,
            regularB = 10,
            MAEK = 30,
            PlotForloop = 500,
            ratio = dict(AE=0,
                        dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
            ),
            # Extra Head (DR project)
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
                enc_forward_w  = [1,   0,    1], # 0802-1 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [4000, 8000,  0], # 0801-1, OK
                    enc_w   = [4000, 8000,  0], # 0801-1, OK
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0], # 0801-1, OK
                ),
            ),
            # Orth
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS= 0,
            # EPOCHS= 14000,
            PlotForloop = 2000,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0,
                inv_Dec=0,
                # inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
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


# ML-AE, 8 layers baseline for MNIST, d=10
def test_3(mode):
    # 0805: ML-AE baseline for MNIST (OK), DR + AE_inv
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET="mnist",
            # BATCHSIZE = 20000,
            # N_dataset = 20000,
            BATCHSIZE = 8000, # 0802-1
            N_dataset = 8000,
            EPOCHS=9000, # 0806-2
            PlotForloop = 1000,
            # regularB = 100,
            # MAEK = 40,
            regularB = 10,
            MAEK = 30,
            ratio = dict(
                        # AE=0.00005, # OK
                        AE=0.0001, # 0806-1, OK
                        # AE=0.0005, # try
                        # AE=0.05,
                        # AE=0.1,
                        dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,    0],
                AE_gradual = [0,  3000,  1],  # [start, end, mode]
            ),
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
            ),
            # Extra Head (DR project)
            # LIS
            LISWeght = dict(
                cross = [1,],
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
                extra_w        = [0,   0,    0],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [10000, 10000,  0],
                    enc_w   = [10000, 10000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 1000,
            ratio = dict(AE=0.0005, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
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


# AE, 5 layers baseline for MNIST, 0806, d=10
def test_4(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000, # 0804-1
            N_dataset = 10000, # 0804-1
            EPOCHS=10000,
            regularB = 100,
            MAEK = 40,
            PlotForloop = 1000,
            ratio = dict(AE=1,
                        dist=0, angle=0, push=0,
                        orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  500,  500,  500,    2000,  10],
                relu             = [    1,    1,    1,    1,      1,   ],
                Enc_require_gard = [    1,    1,    1,    1,      1,   ],
                Dec_require_gard = [    1,    1,    1,    1,      1,   ],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # Extra Head (DR project)
            # LIS
            LISWeght = dict(
                # cross = [1,], # ok
                cross = [0,],
                enc_forward      = [0,    0,    0,    0,      0,    0,   ],
                dec_forward      = [0,    0,    0,    0,      0,    0,   ],
                enc_backward     = [0,    0,    0,    0,      0,    0,   ],
                dec_backward     = [0,    0,    0,    0,      0,    0,   ],
                each             = [0,    0,    0,    0,      0,    0,   ],
                # [dist, angle, push],
                cross_w        = [1,   1,    1],
                enc_forward_w  = [1,   0,    1],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS= 0,
            PlotForloop = 2000,
            ratio = dict(AE=0.05, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  500,  500,  500,    2000,  10],
                relu             = [    1,    1,    1,    1,      1,   ],
                Enc_require_gard = [    0,    0,    0,    0,      0,   ],
                Dec_require_gard = [    1,    1,    1,    1,      1,   ],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,   ],
                dec_forward      = [0,    0,    0,    0,      0,    0,   ],
                enc_backward     = [0,    0,    0,    0,      0,    0,   ],
                dec_backward     = [0,    0,    0,    0,      0,    0,   ],
                each             = [0,    0,    0,    0,      0,    0,   ],
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


# ML-Enc, 8layers baseline for MNIST
def test_5(mode):
    params = base_params(mode)

    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=4000,
            regularB = 10,
            MAEK = 25,
            PlotForloop = 1000,
            ratio = dict(AE=0,
                        dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
            ),
            # Extra Head (DR project)
            # LIS
            LISWeght = dict(
                cross = [1,],
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     1],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,   1,    1],
                enc_forward_w  = [1,   0,    1],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [4000, 8000,  0], # 0801-1, OK
                    enc_w   = [4000, 8000,  0], # 0801-1, OK
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0], # 0801-1, OK
                ),
            ),
            # Orth
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000, # 0806-1
            N_dataset = 10000, # 0806-1
            EPOCHS= 0,
            PlotForloop = 2000,
            ratio = dict(AE=0.005, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0,
                inv_Dec=0,
                # inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # ReLU
            ReluType = dict(
                type = "Leaky", # "InvLeaky"
                Enc_alpha = 0.1,
                Dec_alpha = 0.1,
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
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,   0,    0],
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


# ML-Enc, 8layers baseline for MNIST, d=2
def test_6(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=4000,
            # regularB = 100,
            # MAEK = 40,
            regularB = 10,
            MAEK = 30,
            PlotForloop = 500,
            ratio = dict(AE=0,
                        dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0, inv_Dec=0,
            ),
            # AE layer
            # Extra Head (DR project)
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
                enc_forward_w  = [1,   0,    1], # 0802-1 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [0,   0,    0], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [4000, 8000,  0], # 0801-1, OK
                    enc_w   = [4000, 8000,  0], # 0801-1, OK
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0], # 0801-1, OK
                ),
            ),
            # Orth
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS= 0,
            # EPOCHS= 14000,
            PlotForloop = 2000,
            ratio = dict(AE=1, dist=0, angle=0, push=0,  orth=0, pad=0),
            add_jump = False,
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    1],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                inv_Enc=0,
                inv_Dec=0,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
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
