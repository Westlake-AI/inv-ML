# usps: ML-Enc baseline + Extra-Head + Orth loss

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
                push_gradual = dict( # add 0716: [1 -> 0]
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



# usps, 8 layers Extra + Orth loss, (classs=10), ok
def test_1(mode):
    # copy of ok plan: test_8_0803. USPS, DR + inverse
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=9000,
            regularB = 100,
            # MAEK = 40,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        # orth=0.01,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     40,   10,   10,     10,   10,   8,   4,    ],
                weight           = [       2,    4,    8,      16,   32,   64,  128,  ],
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
                    cross_w = [3000, 10000,  0],
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [5000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [800,  1500, 1],
                each             = [ 15000,  10000,  7000,  4000,  2000, 1000,  400,   ], # 0803-1
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS= 0,
            PlotForloop = 1000,
            MAEK = 30,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
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


# usps, 8 layers Extra + Orth loss, (classs=10), good
def test_2(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="usps",
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=10000,
            # regularB = 100,
            regularB = 10,
            # MAEK = 30,
            MAEK = 20,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10], # 0808-2
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,    2], # 0810-1
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     20,   10,   10,     10,    8,    6,    2,    ],
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
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [4000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5500,  1500,  600,    350,  200,  100,   ],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="usps",
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS= 1,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                # layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
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
                enc_forward_w  = [0,  0,  0],
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




# usps, 8 layers Extra + Orth loss, (classs=10), good inverse
def test_4(mode):
    # copy of ok plan: test_8_0803. USPS
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=9000,
            regularB = 100,
            MAEK = 30,
            PlotForloop = 500,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                #layer           = [256,  256,  256,  256,    256,  256,  256,  256,   3],
                layer            = [0,     40,   10,   10,     10,   10,   10,   10,    ],
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
                enc_forward_w  = [1,   1,   10], # add 0728 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [1,   1,   10],
                extra_w        = [1,   1,    1], # 0730, add new
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    # cross_w = [6000, 7000,  0], # OK, 0731
                    # enc_w   = [6000, 7000,  0], # OK
                    # dec_w   = [0,    0,     0],
                    # each_w  = [0,    0,     0],
                    # extra_w = [6000, 7000,  0], # OK 0731
                    cross_w = [3000, 10000,  0], # 0801-1
                    enc_w   = [4000, 9000,  0], # 0801-1
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [5000, 8000,  0], # 0801-1
                ),
            ),
            # Orth
            OrthWeight = dict(
                #Orth_gradual = [100,  1000, 1], # ori
                Orth_gradual = [800,  1500, 1],
                # layer            = [256,  256,  256,  256,    256,  256,  256,  256,   3],
                each             = [ 15000,  10000,  7000,  4000,  2000, 1000,  400,   ],  # good
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            MAEK = 30,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
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



# usps, 5 layers Extra + Orth loss, (classs=10), good inverse
def test_5(mode):
    # copy of ok plan: test_8_0803. USPS
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS = 9000,
            regularB = 100,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,   10],
                relu             = [    1,    1,    1,    1,      0],
                Enc_require_gard = [    1,    1,    1,    1,      1],
                Dec_require_gard = [    0,    0,    0,    0,      0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     40,   10,   10,     10,   10 ],
                weight           = [       2,    8,    32,     64,   128],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,],
                enc_forward      = [0,    0,    0,    0,      0,    1],
                dec_forward      = [0,    0,    0,    0,      0,    0],
                enc_backward     = [0,    0,    0,    0,      0,    0],
                dec_backward     = [0,    0,    0,    0,      0,    0],
                each             = [0,    0,    0,    0,      0,    0],
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
                    # cross_w = [6000, 7000,  0], # OK, 0731
                    # enc_w   = [6000, 7000,  0], # OK
                    # dec_w   = [0,    0,     0],
                    # each_w  = [0,    0,     0],
                    # extra_w = [6000, 7000,  0], # OK 0731
                    cross_w = [3000, 10000,  0],
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [5000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [800,  1500, 1],
                each             = [  15000,  10000,  4000,  800   ],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            MAEK = 30,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,   10],
                relu             = [    1,    1,    1,    1,      0],
                Enc_require_gard = [    0,    0,    0,    0,      0],
                Dec_require_gard = [    1,    0,    0,    0,      0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,    0,    0,    0,      0,     0],
                dec_forward      = [1,    0,    0,    0,      0,     0],
                enc_backward     = [0,    0,    0,    0,      0,     0],
                dec_backward     = [0,    0,    0,    0,      0,     0],
                each             = [0,    0,    0,    0,      0,     0],
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


# usps, 8 layers Extra + Orth loss, (classs=10), ok
def test_6(mode):
    # copy plan: test_8_0803. USPS
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=9000,
            # regularB = 100, # ori
            regularB = 10,
            MAEK = 30,
            PlotForloop = 500,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     40,   20,   10,     10,    8,    4,    2,    ], # 0807-2
                weight           = [       2,    4,    8,      16,   32,   64,  128,   ], # try@, 0729-2, test_4
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
                enc_forward_w  = [1,   1,   10], # add 0728 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                extra_w        = [1,   1,    4], # 0807-1
                # gradual
                LIS_gradual = [0,    0,     1],
                push_gradual = dict(
                    cross_w = [3000, 10000,  0],
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [5000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [800,  1500, 1],
                # each             = [ 15000,  10000,  7000,  4000,  2000, 1000,  400,   ],  # 0803-1, good
                each             = [ 15000,  10000,  8000,  4000,  2000,  1000,  400,  ],  # 0807-2, try
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            MAEK = 30,
            EPOCHS= 0,
            PlotForloop = 1,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
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


# usps, 8 layers Extra + Orth loss, (classs=10), ok
def test_7(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=9000,
            # regularB = 100,
            regularB = 10,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     40,   10,   10,     10,   10,   8,   4,    ],
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
                each_w         = [1,   1,   10],
                extra_w        = [1,   1,    1],
                # gradual
                LIS_gradual = [0,    0,     1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [3000, 10000,  0],
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [5000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                #Orth_gradual = [100,  1000, 1], # 0731-1, ori
                Orth_gradual = [800,  1500, 1], # 0731-2, try
                each             = [ 15000,  10000,  7000,  4000,  2000, 1000,  400,   ],  # good
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            DATASET = "usps",
            numberClass = 10,
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS= 0,
            PlotForloop = 1,
            MAEK = 30,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                Dec_require_gard = [    1,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,    0,    0,    0,      0,    0,    0,    0,     0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
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
                enc_forward_w  = [0,  0,  0],
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


# usps, 8 layers Extra + Orth loss, (classs=10), good
def test_3(mode):
    params = base_params(mode)
    if mode == 'encoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="usps",
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS=10000,
            # regularB = 3,
            regularB = 100,
            # MAEK = 40,
            MAEK = 30,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.1,
                        pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10], # 0808-2, try
                relu             = [    1,    1,    1,    1,      1,    1,    1,    0],
                Enc_require_gard = [    1,    1,    1,    1,      1,    1,    1,    1],
                Dec_require_gard = [    0,    0,    0,    0,      0,    0,    0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,     20,   10,   10,     10,   10,    6,    2,   ],
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
                    cross_w = [4000, 9000,  0],
                    enc_w   = [4000, 9000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [4000, 8000,  0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [50,  1200, 1],
                each             =   [8000,  5500,  1500,  600,    350,  200,  100,   ],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            numberClass = 10,
            DATASET="usps",
            BATCHSIZE = 9298,
            N_dataset = 9298,
            EPOCHS= 1,
            PlotForloop = 1,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [256,  256,  256,  256,    256,  256,  256,  256,   10],
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
                enc_forward_w  = [0,  0,  0],
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                extra_w        = [1,   1,    1], # 0730, add new
                # gradual
                LIS_gradual = [0,  0,  1],
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
