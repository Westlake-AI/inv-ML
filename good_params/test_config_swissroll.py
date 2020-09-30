# Config for Swiss Roll, 0827

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


def base_params(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            BATCHSIZE = 800,
            N_dataset = 800,
            EPOCHS=4000,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8, orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,     50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project): None
            ExtraHead = dict(
                layer            = [],
                weight           = [],
            ),
            # AE: None
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                AE_gradual = [0,  0,  1],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,      0,   0,   0,   0,    0],
                # [dist, angle, push]
                cross_w        = [1,   1,    1],
                enc_forward_w  = [0,   0,    0], 
                dec_forward_w  = [0,   0,    0],
                enc_backward_w = [0,   0,    0],
                dec_backward_w = [0,   0,    0],
                each_w         = [0,   0,    0],
                # gradual
                LIS_gradual    = [0,   0,    1],
                push_gradual = dict( # [1 -> 0]
                    cross_w    = [500,  1000,  0],
                    enc_w      = [500,  1000,  0],
                    dec_w      = [0,    0,     0],
                    each_w     = [0,    0,     0],
                    extra_w    = [500,  0,     0],
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
                padding          =  [     0,    0,   0,   0,     0,   0,    0,   0,   0],
                pad_w            =  [     0,    0,   0,   0,     0,   0,    0,   0,   0],
                pad_gradual = [0,   0,   1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 800,
            N_dataset = 800,
            EPOCHS= 1000,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0, pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [0,    500,   0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param



# swiss roll, 8layer, baseline + Extra-Head
def test_1_old(mode):
    params = base_params(mode)
    if mode == "encoder":
        param = dict(
            # BATCHSIZE = 1200,
            # N_dataset = 1200,
            EPOCHS=4000,
            PlotForloop = 1000,
            # regular
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  6,   5,   4,   3,     2,   2,   2,   2,    0],
                weight           = [    20,  30,  40,  55,     75,  100,  130, 165, 200],
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,  1e-1,  4e-1,  1],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push]
                cross_w        = [1,  1,    1],
                enc_forward_w  = [1,  0,    1], # ori
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],
                push_gradual = dict( # add 0716: [1 -> 0], Ok Plan
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],  # @, ok
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0], # ok, prev
                ),
            ),
        )
    elif mode == "decoder":
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2], # ori
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0], # 0715.1
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    params.update(param)
    return params


# swiss roll, 8layer, baseline + Extra-Head
def test_1(mode):
    params = base_params(mode)
    if mode == "encoder":
        param = dict(
            BATCHSIZE = 1200,
            N_dataset = 1200,
            EPOCHS=4000,
            # regular
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0, pad=0),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  6,   5,   4,   3,     2,   2,   2,   2,    0],
                weight           = [    20,  30,  40,  55,     75,  100,  130, 165, 200], # @ Plan B
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,  1e-1,  4e-1,  1],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push]
                cross_w        = [1,  1,    1],
                enc_forward_w  = [1,  0,    1], # ori
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],
                push_gradual = dict( # add 0716: [1 -> 0], Ok Plan
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0], # ok, prev
                ),
            ),
        )
    elif mode == "decoder":
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2], # ori
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0], # 0715.1
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    params.update(param)
    return params


# swiss roll, 8layer, baseline + Extra-Head
def test_2(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            PlotForloop = 1000,
            EPOCHS=4000, # now
            ratio = dict(AE=0, dist=1, angle=0, push=0.8, orth=0, pad=0),  # baseline
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,  3,   3,   2,     2,   2,   2,   2,   0],
                weight           = [    120, 75,  110, 50,    70,  90,  100, 150,  200]
            ),
            # AE layer: None
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0, 0.8, 0.7, 0.6, 0.5,  0.4, 0.3, 0.2, 0.1,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [1,  0,    1],  # 0716.1,
                enc_forward_w  = [1e-3, 0, 1e-2],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0],
                ),
            ),
            # Orth: None
            # inverse mode: None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0, # ori
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716.1, add
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# swiss roll, 8layer, baseline + Extra-Head + Orth
def test_3(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            # EPOCHS=5000, # ok
            EPOCHS=6000,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.005,
                        # pad=0),
                        pad=0.1),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,  3,   3,   2,     2,   2,   2,   2,   0],
                weight           = [    120, 80,  120, 60,    80,  90,  120, 150,  200]
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push],
                cross_w        = [1,  0,    1],
                enc_forward_w  = [1,  0,    1],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
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
                Orth_gradual = [0, 5000, 1], # 0717.1
                each             = [       10,  10,  5,   5,     2,    0.5,  0.1],  # 0716.3, ok
            ),
            # inverse mode, None, remove
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1", # ok
                padding          =  [     0,   48,  45,  40,    35,  15,   15,  10,  10],
                pad_w            =  [     0,    1,   2,   3,     4,   8,   16,  24,  32],
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# swiss roll, 8layer, baseline + Extra-Head + Orth,
def test_4(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            EPOCHS=6000, # 0826-1
            # EPOCHS=4000,
            PlotForloop = 1000,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,
                        orth=0.005,  pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,   3,   3,   2,     2,   2,   2,   2,    0  ], # 0826-1-1, try
                weight           = [    30,  50,  60,  70,    80,  100, 120, 150,  200], # 0826-1-1, try
                push_w           = [  1e-1,  1,   2,   4,     6,   8,  12,   16,   24 ], # 0826-2-1, try
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push], add 0716
                cross_w        = [1,  0,    1],  # 0716
                enc_forward_w  = [1,  0,    1],  # 0716
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0], # ok
                    enc_w   = [500,  1000,  0], # ok
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [400,  900,   0],
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [0, 5000, 1],
                each             = [       12,  10,  6,   5,     2,    0.8,  0.1],
            ),
            # inverse mode, None
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],  # 0716
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param



# swiss roll, 8layer, baseline + Extra-Head + Orth + Padding
def test_5(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            EPOCHS=5000,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.005,
                        pad=0.1),
                        # pad=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,   3,   3,   2,     2,   2,   2,   2,    0  ],
                weight           = [    30,  50,  60,  70,    80,  100, 120, 150,  200],
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push]
                cross_w        = [1,  0,    1],
                enc_forward_w  = [1,  0,    1],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
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
                Orth_gradual = [0, 5000, 1],
                each             = [       12,  10,  6,   5,     2,    0.8,  0.1],
            ),
            # inverse mode, None
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,   45,  40,  35,    30,  25,   12,   5,   2],
                pad_w            =  [     0,    1,   2,   3,     4,   8,   12,  16,  24],
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,    0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,    0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# swiss roll, 8layer,  baseline + Extra-Head + Orth + Padding
def test_6(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            EPOCHS=6000,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.005,
                        pad=0.1),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,   3,   3,   2,     2,   2,   2,   2,    0  ], # 0826
                weight           = [    30,  50,  60,  70,    80,  100, 120, 150,  200], # 0826
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push], add 0716
                cross_w        = [1,  0,    1],
                enc_forward_w  = [1,  0,    1],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0], # ok
                    enc_w   = [500,  1000,  0], # ok
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0], # Good
                    # extra_w = [400,  900,   0], # 0826, try
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [0, 5000, 1],
                # each             = [       15,  12,  7,   5,     2,    0.8,  0.1],  # 0825-2, try
                each             = [       12,  10,  6,   5,     2,    0.8,  0.1],  # 0827, ok
            ),
            # inverse mode
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                padding          =  [     0,   45,  40,  30,    20,  15,   10,   8,   8], # 0827
                pad_w            =  [     0,    1,   2,   3,     4,   8,   16,  24,  32],  # 0827
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push], add 0716
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# swiss roll, 8layer, baseline + Extra-Head + Orth + Padding
def test_7(mode):
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            # BATCHSIZE = 5000,
            # N_dataset = 5000,
            EPOCHS=5000, # 0826-1
            # EPOCHS=4000,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.005,
                        pad=0.1),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2], # ori
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0], # 0715
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,   3,   3,   2,     2,   2,   2,   2,    0  ], # 0826-1
                weight           = [    30,  50,  60,  70,    80,  100, 120, 150,  200], # 0826-1
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push]
                cross_w        = [1,  0,    1],
                enc_forward_w  = [1,  0,    1],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0], # ok
                    enc_w   = [500,  1000,  0], # ok
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0], # Good
                    # extra_w = [400,  900,   0], # 0826
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [0, 5000, 1],
                each             = [       12,  10,  6,   5,     2,    0.8,  0.1],  # 0825
            ),
            # inverse mode, None
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1",
                # padding          =  [     0,   45,  40,  35,    30,  25,   15,  10,  10], # 0827-7
                padding          =  [     0,   45,  40,  35,    30,  25,    6,   4,   2], # 0827-7, ok
                pad_w            =  [     0,    1,   2,   3,     4,   8,   12,  16,  24],
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,    0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,    0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,    0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict(
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param


# swiss roll, 8layer, baseline + Extra-Head + Orth + Padding, 0826
def test_8(mode):
    # baseline + Extra-Head + Orth (nearly best for inverse)
    param = {}
    if mode == 'encoder':
        param = dict(
            # regular
            # BATCHSIZE = 5000,
            # N_dataset = 5000,
            EPOCHS=6000,
            PlotForloop = 1000,
            ratio = dict(AE=0, dist=1, angle=0, push=0.8,
                        orth=0.005,
                        pad=0.1),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   1,   1,   1,   1,   1,     1,   1,   1,   1],
                Dec_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # Extra Head (DR project)
            ExtraHead = dict(
                layer            = [0,  3,   3,   3,   2,     2,   2,   2,   2,    0  ], # 0826, try
                weight           = [    30,  50,  60,  70,    80,  100, 120, 150,  200], # 0826, try
            ),
            # AE layer (None)
            AEWeight = dict(
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [1,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # each loss: [dist, angle, push]
                cross_w        = [1,  0,    1],
                enc_forward_w  = [1,  0,    1],
                dec_forward_w  = [0,  0,    0],
                enc_backward_w = [0,  0,    0],
                dec_backward_w = [0,  0,    0],
                each_w         = [0,  0,    0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add 0716: [1 -> 0]
                    cross_w = [500,  1000,  0],
                    enc_w   = [500,  1000,  0],
                    dec_w   = [0,    0,     0],
                    each_w  = [0,    0,     0],
                    extra_w = [500,  1000,  0], # Good
                    # extra_w = [400,  900,   0], # 0826-1-1, try
                ),
            ),
            # Orth
            OrthWeight = dict(
                Orth_gradual = [0, 5000, 1],
                # each             = [       15,  12,  7,   5,     2,    0.8,  0.1],
                each             = [       12,  10,  6,   5,     2,    0.8,  0.1],  # 0825
            ),
            # inverse mode, None
            InverseMode = dict(
                mode = "ZeroPadding",
                loss_type = "L1", # ok
                padding          =  [     0,   45,  40,  35,    25,  15,   15,  10,   8],
                pad_w            =  [     0,    1,   2,   3,     4,   8,   16,  24,  32],
                pad_gradual = [0,  3500, 1],
            ),
        )
    elif mode == 'decoder':
        param = dict(
            # regular
            BATCHSIZE = 1000,
            N_dataset = 1000,
            EPOCHS= 0,
            ratio = dict(AE=0.005, dist=1, angle=0, push=0.8,  orth=0),
            # structure
            NetworkStructure = dict(
                layer            = [3,  50,  50,  50,  50,    50,  50,  50,  50,   2],
                relu             = [   0,   1,   1,   1,   1,     1,   1,   1,   0],
                Enc_require_gard = [   0,   0,   0,   0,   0,     0,   0,   0,   0],
                Dec_require_gard = [   1,   0,   0,   0,   0,     0,   0,   0,   0],
                inv_Enc=0, inv_Dec=1,
            ),
            # AE layer
            AEWeight = dict(
                each             = [1,   0,   0,   0,   0,     0,   0,   0,   0,   0],
                AE_gradual = [0, 0, 1],  # [start, end, mode]
            ),
            # LIS
            LISWeght = dict(
                cross = [0,], # ok
                enc_forward      = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_forward      = [1,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                enc_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                dec_backward     = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                each             = [0,  0,   0,   0,   0,     0,   0,   0,   0,    0],
                # [dist, angle, push],
                cross_w        = [0,  0,  0],
                enc_forward_w  = [0,  0,  0],  # [dist, angle, push]
                dec_forward_w  = [0,  0,  0],
                enc_backward_w = [0,  0,  0],
                dec_backward_w = [0,  0,  0],
                each_w         = [0,  0,  0],
                # gradual
                LIS_gradual = [0,  0,  1],  # [start, end, mode]
                push_gradual = dict( # add
                    cross_w = [0,    0,     0],
                    enc_w   = [0,    0,     0],
                    dec_w   = [500,  1000,  0],
                    each_w  = [0,    0,     0],
                    extra_w = [0,    0,     0],
                ),
            ),
        )
    # result
    return param

