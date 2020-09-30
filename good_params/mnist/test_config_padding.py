# mnist baseline + ExtraHead + Orth Loss + Padding loss

def import_test_config(test_num, mode='encoder'):
    if test_num == 1:
        return test_1(mode)
    elif test_num == 2:
        return test_2(mode)
    elif test_num ==3:
        return test_3(mode)
    elif test_num ==4:
        return test_4(mode)
    # elif test_num ==5:
    #     return test_5(mode)
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


# mnist, 8layers ML-Enc + ExtraHead + Orth + Padding, d=2, good
def test_1(mode):
    # copy of 0812: 8 layers inverse + DR + padding, (classs=10), (0812-2)
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
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        # pad=20
                        pad=100 # ok
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
            # inverse mode, 0808-1
            InverseMode = dict(
                mode = "ZeroPadding", #"CSinverse",
                loss_type = "L1",
                padding          =  [     0,  780, 760, 700,   750,  650, 700, 630, 600], # 0814
                pad_w            =  [     0,    2,   4,   16,     8,   32,   16,  64,  64],  # 0813
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
                cross = [0,], # ok
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


# mnist, 8layers ML-Enc + ExtraHead + Orth + Padding, d=2, ok
def test_2(mode):
    # copy of test_6_0814_6: 8 layers inverse + DR + padding, Good!
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=10000,
            regularB = 100,
            # regularB = 10,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        # pad=20
                        pad=100
                    ),
            # structure
            NetworkStructure = dict(
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
                LIS_gradual = [0,    0,     1], # [start, end, mode]
                push_gradual = dict(
                    cross_w = [4000, 9500,  0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 8000,  0],
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
                padding          =  [     0,  780, 760, 700,   750,  650, 700, 630, 600], # 0814, good for mnist!
                # pad_w            =  [     0,    1,   1,   1,     1,   1,    1,   1,   1],  # ori
                pad_w            =  [     0,    2,   4,   16,     8,   32,   16,  64,  64],  # 0813-3,
                # pad_gradual = [0,  3500, 1],  # ori
                pad_gradual = [0,  3000, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,  # 0812
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


# mnist, 8layers ML-Enc + ExtraHead + Orth + Padding, d=10
def test_3(mode):
    # copy of test_6_0814_6: 8 layers inverse + DR + padding, Good!
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS=10000,
            # regularB = 100,
            regularB = 10,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=100
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
                # enc_forward_w  = [1,   1,   10],
                enc_forward_w  = [1,   1,   15],
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1], # [start, end, mode]
                push_gradual = dict(
                    cross_w = [4000, 9500,  0],
                    enc_w   = [3000, 8000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [2000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5800,  1500,  600,    380,  240,  100,   ],
            ),
            # inverse mode, 0808-1
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,  780, 760, 700,   750,  650, 700, 630, 600], # 0814, good for mnist
                pad_w            =  [     0,    2,   4,   16,     8,   32,   16,  64,  64],
                pad_gradual = [0,  3000, 1],
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


# achieve more sparse embedding:
# mnist, 8layers ML-Enc + ExtraHead + Orth + Padding, d=10, train last 2 layer for more sparse.
def test_4(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            resume_from = "./final_summary/mnist/+Padding/test1_good/MLP_8layer_0814_6_mnist(inv+pad)_good/checkpoints/8000.pth",
            numberClass = 10,
            # regular
            DATASET="mnist",
            BATCHSIZE = 10000,
            N_dataset = 10000,
            EPOCHS = 4000,
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad = 0,
                        # pad=20
                        # pad=100 # ok
                    ),
            # structure
            NetworkStructure = dict(
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,   10],
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2], # prev
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                # Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1], # prev
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                # layer            = [0,     12,   10,   8,      6,    4,    3,    2,    ], # prev
                # weight           = [       2,    4,    8,      16,   32,   64,  128,   ],
                layer            = [0,     0,    0,    0,      0,    0,    0,    2,    ],
                weight           = [                                             64,   ],
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
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1],
                # push_gradual = dict(
                #     cross_w = [4000, 9000,  0],
                #     enc_w   = [3000, 8000,  0],
                #     dec_w   = [0,    0,     0],
                #     each_w  = [0,    0,     0],
                #     extra_w = [3000, 8000,  0],
                # ),
                push_gradual = dict(
                    cross_w = [2000, 8000,  0],
                    enc_w   = [ 500, 6000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [ 500, 6000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                # Orth_gradual = [50,  1200, 1], # prev
                # each             =   [8000,  5500,  1500,  600,    350,  200,  100,   ],
                Orth_gradual = [0,   500,   1],
                each             =   [0,      0,    0,    0,       0,     0,   400,   ],
            ),
            # inverse mode, 0808-1
            # InverseMode = dict(
            #     mode = "ZeroPadding", #"CSinverse",
            #     loss_type = "L1",
            #     # padding          =  [     0,  780, 760, 700,   750,  650, 700, 630, 600], # prev
            #     # pad_w            =  [     0,    2,   4,   16,     8,   32,   16,  64,  64],  # prev
            #     # pad_gradual = [0,  3500, 1], # prev
            #     padding          =  [     0,    0,   0,    0,     0,    0,    0,   550],
            #     pad_w            =  [     0,    0,   0,    0,     0,    0,    0,   32],
            #     pad_gradual = [0,   500,  1],
            # ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="mnist",
            BATCHSIZE = 10000, 
            N_dataset = 10000,
            # EPOCHS= 0, # prev
            EPOCHS= 1000, # try
            regularB = 3,
            MAEK = 15,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0,
                        # push=0.8, # prev
                        push = 5,
                        orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [784,  784,  784,  784,    784,  784,  784,  784,    2],
                layer            = [784,  784,  784,  784,    784,  784,  784,  784,    10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                # Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0], # prev
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    1], # try
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0,   0,   1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                # cross = [0,], # prev
                cross = [1,], # try
                enc_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_forward      = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                enc_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                dec_backward     = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                each             = [0,    0,    0,    0,      0,    0,    0,    0,     0],
                # [dist, angle, push],
                cross_w        = [1,  1,  10],
                enc_forward_w  = [1,  0,  10],
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],
                push_gradual = dict(
                    # cross_w = [0,    0,     0], # prev
                    cross_w = [500,  5000,  0], # try
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
# layer 14: D_rank <= 150
